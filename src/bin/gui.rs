//! Qwen3-TTS Desktop GUI — egui native application
//!
//! Usage:
//!   cargo build --features gui,cuda --bin gui
//!   cargo run  --features gui,cuda --bin gui

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use eframe::egui;
use egui::{Color32, ProgressBar as EguiProgressBar, RichText, ScrollArea, TextEdit};

use qwen3_tts::i18n::{Locale, Tr, SPEAKERS};
use qwen3_tts::{parse_device, AudioBuffer, Qwen3TTS, SynthesisOptions};

// ── Shared progress between UI thread and background worker ─────────────────────

#[derive(Clone, Default)]
struct Progress {
    phase: String, // idle | loading | generating | saving | done | error
    elapsed_secs: f64,
    #[allow(dead_code)]
    current_frame: usize,
    result_path: Option<PathBuf>,
    error_msg: String,
}

// ── History entry ──────────────────────────────────────────────────────────────

#[derive(Clone)]
struct HistoryEntry {
    #[allow(dead_code)]
    label: String,
    path: PathBuf,
    duration_secs: f32,
    #[allow(dead_code)]
    frames: usize,
}

// ── Main app ───────────────────────────────────────────────────────────────────

struct TtsApp {
    // Inputs
    text: String,
    speaker_idx: usize,
    lang_idx: usize,
    duration: f64,
    seed: u64,
    temperature: f64,
    top_k: u32,
    top_p: f64,
    rep_penalty: f64,
    model_dir: String,
    output_dir: String,

    // UI settings
    locale: Locale,

    // Batch queue
    batch: Vec<String>,

    // Background worker
    progress: Arc<Mutex<Progress>>,
    worker: Option<thread::JoinHandle<()>>,

    // History
    history: Vec<HistoryEntry>,

    // Audio playback
    #[allow(dead_code)]
    currently_playing: Option<usize>,
}

impl Default for TtsApp {
    fn default() -> Self {
        Self {
            text: String::new(),
            speaker_idx: 0,
            lang_idx: 0,
            duration: 10.0,
            seed: 42,
            temperature: 0.7,
            top_k: 50,
            top_p: 0.9,
            rep_penalty: 1.05,
            model_dir: "model".into(),
            output_dir: "output/voice".into(),
            locale: Locale::ZhTw,
            batch: Vec::new(),
            progress: Arc::new(Mutex::new(Progress::default())),
            worker: None,
            history: Vec::new(),
            currently_playing: None,
        }
    }
}

impl TtsApp {
    fn tr(&self) -> Tr {
        Tr::new(self.locale)
    }

    /// Spawn a background thread that loads the model and generates audio.
    fn start_generation(&mut self) {
        // Don't start if already running
        if self.worker.is_some() {
            return;
        }
        let text = if self.batch.is_empty() {
            self.text.clone()
        } else {
            self.batch.remove(0)
        };
        if text.trim().is_empty() {
            return;
        }

        let progress = self.progress.clone();
        let model_dir = self.model_dir.clone();
        let output_dir = self.output_dir.clone();
        let speaker = SPEAKERS[self.speaker_idx].to_string();
        let langs = self.tr().languages();
        let language = langs[self.lang_idx].internal.to_string();
        let duration = self.duration;
        let seed = self.seed;
        let temperature = self.temperature;
        let top_k = self.top_k;
        let top_p = self.top_p;
        let rep_penalty = self.rep_penalty;

        // Reset progress
        *progress.lock().unwrap() = Progress::default();

        self.worker = Some(thread::spawn(move || {
            let result = run_generation(
                &text,
                &model_dir,
                &output_dir,
                &speaker,
                &language,
                duration,
                seed,
                temperature,
                top_k as usize,
                top_p,
                rep_penalty,
                &progress,
            );
            let mut p = progress.lock().unwrap();
            match result {
                Ok(audio) => {
                    p.phase = "done".into();
                    p.result_path = Some(audio.path);
                }
                Err(e) => {
                    p.phase = "error".into();
                    p.error_msg = format!("{:#}", e);
                }
            }
        }));
    }

    /// Play a WAV file in a background thread (spawns its own audio stream).
    fn play_audio(&mut self, idx: usize) {
        if let Some(entry) = self.history.get(idx) {
            self.currently_playing = Some(idx);
            let path = entry.path.clone();
            thread::spawn(move || {
                if let Ok((_stream, handle)) = rodio::OutputStream::try_default() {
                    if let Ok(file) = std::fs::File::open(&path) {
                        if let Ok(source) = rodio::Decoder::new(std::io::BufReader::new(file)) {
                            if let Ok(sink) = rodio::Sink::try_new(&handle) {
                                sink.append(source);
                                sink.sleep_until_end();
                            }
                        }
                    }
                }
            });
        }
    }

    /// Check if worker finished and move result to history.
    fn poll_worker(&mut self) {
        if let Some(handle) = self.worker.take() {
            if handle.is_finished() {
                // Worker done — consume result
                handle.join().ok();
                let p = self.progress.lock().unwrap();
                if p.phase == "done" {
                    if let Some(ref path) = p.result_path {
                        // Estimate frames from duration
                        // (we don't have exact frame count here; re-save avoids extra dep)
                        let label = path
                            .file_stem()
                            .map(|s| s.to_string_lossy().into_owned())
                            .unwrap_or_default();
                        let duration_secs = AudioBuffer::load(path)
                            .ok()
                            .map(|a| a.duration())
                            .unwrap_or(0.0);
                        self.history.push(HistoryEntry {
                            label,
                            path: path.clone(),
                            duration_secs,
                            frames: 0, // not tracked from high-level API
                        });
                    }
                }
            } else {
                // Still running — put it back
                self.worker = Some(handle);
            }
        }
    }

    /// True when a generation is in progress.
    fn is_busy(&self) -> bool {
        self.worker.as_ref().map_or(false, |h| !h.is_finished())
    }

    /// Generate a timestamp-based filename.
    fn output_filename() -> String {
        let secs = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_secs();
        format!("voice_{}.wav", secs)
    }
}

// ── Generation runner (background thread) ─────────────────────────────────────

struct GeneratedAudio {
    path: PathBuf,
}

fn run_generation(
    text: &str,
    model_dir: &str,
    output_dir: &str,
    speaker: &str,
    language: &str,
    duration: f64,
    seed: u64,
    temperature: f64,
    top_k: usize,
    top_p: f64,
    rep_penalty: f64,
    progress: &Arc<Mutex<Progress>>,
) -> anyhow::Result<GeneratedAudio> {
    // Phase 1: loading model
    {
        let mut p = progress.lock().unwrap();
        p.phase = "loading".into();
        p.elapsed_secs = 0.0;
    }
    let t0 = Instant::now();

    let device = parse_device("auto")?;
    let _device_name = qwen3_tts::device_info(&device);

    let model = Qwen3TTS::from_pretrained(model_dir, device)?;

    let spk: qwen3_tts::Speaker = speaker.parse()?;
    let lang: qwen3_tts::Language = language.parse()?;

    let max_frames = (duration * 12.5) as usize;

    let options = SynthesisOptions {
        max_length: max_frames,
        temperature,
        top_k,
        top_p,
        repetition_penalty: rep_penalty,
        seed: Some(seed),
        ..Default::default()
    };

    // Phase 2: generating
    {
        let mut p = progress.lock().unwrap();
        p.phase = "generating".into();
        p.elapsed_secs = t0.elapsed().as_secs_f64();
    }

    let audio = model.synthesize_with_voice(text, spk, lang, Some(options))?;

    // Phase 3: saving
    {
        let mut p = progress.lock().unwrap();
        p.phase = "saving".into();
    }

    let filename = TtsApp::output_filename();
    let output_path = PathBuf::from(output_dir).join(&filename);
    if let Some(parent) = output_path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    audio.save(&output_path)?;

    let elapsed = t0.elapsed().as_secs_f64();

    // Update final progress
    {
        let mut p = progress.lock().unwrap();
        p.phase = "done".into();
        p.elapsed_secs = elapsed;
        p.result_path = Some(output_path.clone());
    }

    Ok(GeneratedAudio { path: output_path })
}

// ── egui App implementation ────────────────────────────────────────────────────

impl eframe::App for TtsApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Poll background worker
        self.poll_worker();
        let busy = self.is_busy();
        let tr = self.tr();

        // Request continuous repaint while busy
        if busy {
            ctx.request_repaint_after(Duration::from_millis(50));
        }

        // ── Top bar: paths and language switcher ─────────────────────────────
        egui::TopBottomPanel::top("top_bar").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label(RichText::new(tr.model_label()).color(Color32::from_rgb(180, 180, 180)));
                ui.text_edit_singleline(&mut self.model_dir);
                ui.separator();
                ui.label(RichText::new(tr.output_label()).color(Color32::from_rgb(180, 180, 180)));
                ui.text_edit_singleline(&mut self.output_dir);

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(tr.ui_language_label());
                    egui::ComboBox::from_id_salt("ui_lang")
                        .selected_text(self.locale.label())
                        .show_ui(ui, |ui| {
                            for &(l, label, _code) in Locale::variants() {
                                if ui.selectable_label(self.locale == l, label).clicked() {
                                    self.locale = l;
                                }
                            }
                        });
                });
            });
        });

        // ── Status bar at bottom ────────────────────────────────────────────
        egui::TopBottomPanel::bottom("status_bar").show(ctx, |ui| {
            let p = self.progress.lock().unwrap();
            match p.phase.as_str() {
                "loading" => {
                    ui.colored_label(Color32::YELLOW, tr.loading_model(p.elapsed_secs));
                }
                "generating" => {
                    ui.colored_label(Color32::YELLOW, tr.generating(p.elapsed_secs));
                }
                "saving" => {
                    ui.colored_label(Color32::YELLOW, tr.saving());
                }
                "done" => {
                    let name = p
                        .result_path
                        .as_ref()
                        .and_then(|p| p.file_name())
                        .map(|n| n.to_string_lossy())
                        .unwrap_or_default();
                    ui.colored_label(Color32::GREEN, tr.done(&name));
                }
                "error" => {
                    ui.colored_label(Color32::RED, tr.error(&p.error_msg));
                }
                _ => {
                    let count = self.history.len();
                    ui.label(tr.files_generated(count));
                }
            }
        });

        // ── Main content ────────────────────────────────────────────────────
        egui::CentralPanel::default().show(ctx, |ui| {
            ScrollArea::vertical().show(ui, |ui| {
                // ── Text input ────────────────────────────────────────────
                ui.heading(tr.text_input_heading());
                ui.add_sized(
                    egui::vec2(ui.available_width(), 120.0),
                    TextEdit::multiline(&mut self.text)
                        .hint_text(tr.text_input_hint())
                        .desired_width(f32::INFINITY),
                );

                ui.horizontal(|ui| {
                    let add_label = if self.batch.is_empty() {
                        tr.queue_batch().to_string()
                    } else {
                        tr.queue_batch_count(self.batch.len())
                    };
                    if ui.button(add_label).clicked() && !self.text.trim().is_empty() {
                        self.batch.push(self.text.trim().to_string());
                        self.text.clear();
                    }
                    if ui.button(tr.clear_batch()).clicked() {
                        self.batch.clear();
                    }
                    if !self.batch.is_empty() {
                        ui.separator();
                        ui.label(tr.next_label());
                        for t in &self.batch {
                            let s: String = t.chars().take(30).collect();
                            ui.label(format!("\u{2022} {s}\u{2026}"));
                        }
                    }
                });

                ui.add_space(8.0);

                // ── Parameters ───────────────────────────────────────────
                ui.heading(tr.params_heading());
                egui::Grid::new("params")
                    .num_columns(4)
                    .striped(true)
                    .min_col_width(80.0)
                    .show(ui, |ui| {
                        // Row 1: Speaker + Language
                        ui.label(tr.speaker_label());
                        egui::ComboBox::from_id_salt("speaker_combo")
                            .selected_text(SPEAKERS[self.speaker_idx])
                            .show_ui(ui, |ui| {
                                for (i, s) in SPEAKERS.iter().enumerate() {
                                    let name = format!("{}{}", s[..1].to_uppercase(), &s[1..]);
                                    ui.selectable_value(&mut self.speaker_idx, i, name);
                                }
                            });
                        ui.label(tr.lang_label());
                        let langs = tr.languages();
                        egui::ComboBox::from_id_salt("lang_combo")
                            .selected_text(langs[self.lang_idx].display)
                            .show_ui(ui, |ui| {
                                for (i, lang) in langs.iter().enumerate() {
                                    ui.selectable_value(&mut self.lang_idx, i, lang.display);
                                }
                            });
                        ui.end_row();

                        // Row 2: Duration + Seed
                        ui.label(tr.duration_label());
                        ui.add(egui::Slider::new(&mut self.duration, 1.0..=120.0));
                        ui.label(tr.seed_label());
                        ui.add(egui::DragValue::new(&mut self.seed).range(0..=u64::MAX));
                        ui.end_row();

                        // Row 3: Temperature + Top-K
                        ui.label(tr.temperature_label());
                        ui.add(egui::Slider::new(&mut self.temperature, 0.0..=2.0));
                        ui.label(tr.topk_label());
                        ui.add(egui::DragValue::new(&mut self.top_k).range(1..=200));
                        ui.end_row();

                        // Row 4: Top-P + Rep Penalty
                        ui.label(tr.topp_label());
                        ui.add(egui::Slider::new(&mut self.top_p, 0.0..=1.0));
                        ui.label(tr.rep_penalty_label());
                        ui.add(egui::Slider::new(&mut self.rep_penalty, 1.0..=2.0));
                        ui.end_row();
                    });

                ui.add_space(12.0);

                // ── Generate button ──────────────────────────────────────
                ui.horizontal(|ui| {
                    let btn_text: String = if busy {
                        tr.generating_text().into()
                    } else if !self.batch.is_empty() {
                        tr.generate_queued(self.batch.len())
                    } else {
                        tr.generate().to_string()
                    };

                    let btn = egui::Button::new(RichText::new(&btn_text).size(18.0))
                        .min_size(egui::vec2(180.0, 40.0));
                    if ui.add_enabled(!busy, btn).clicked() {
                        self.start_generation();
                    }
                });

                // ── Inline progress ──────────────────────────────────────
                {
                    let p = self.progress.lock().unwrap();
                    if p.phase != "idle" && p.phase != "done" && p.phase != "error" {
                        ui.add_space(4.0);
                        ui.add(
                            EguiProgressBar::new(0.5) // indeterminate
                                .animate(true)
                                .text(match p.phase.as_str() {
                                    "loading" => tr.loading_inline(p.elapsed_secs),
                                    "generating" => tr.generating_inline(p.elapsed_secs),
                                    "saving" => tr.saving_inline().into(),
                                    _ => p.phase.clone(),
                                }),
                        );
                    }
                }

                ui.add_space(16.0);

                // ── History ──────────────────────────────────────────────
                if !self.history.is_empty() {
                    ui.heading(tr.history_heading());
                    let mut play_idx: Option<usize> = None;

                    for (rev_i, entry) in self.history.iter().rev().enumerate() {
                        let file_name = entry
                            .path
                            .file_name()
                            .map(|n| n.to_string_lossy())
                            .unwrap_or_default();
                        let dur = entry.duration_secs;

                        // rev_i counts from 0 (most recent); convert to original index
                        let orig_i = self.history.len() - 1 - rev_i;

                        ui.horizontal(|ui| {
                            ui.label(
                                RichText::new(format!("{file_name}"))
                                    .color(Color32::from_rgb(200, 200, 200)),
                            );
                            ui.label(format!("({dur:.2}s)"));
                            if ui.button(tr.play_btn()).clicked() {
                                play_idx = Some(orig_i);
                            }
                            if ui.button(tr.open_btn()).clicked() {
                                if let Some(parent) = entry.path.parent() {
                                    #[cfg(target_os = "windows")]
                                    let _ =
                                        std::process::Command::new("explorer").arg(parent).spawn();
                                }
                            }
                        });
                    }
                    if let Some(idx) = play_idx {
                        self.play_audio(idx);
                    }
                }
            });
        });
    }
}

// ── Font setup ─────────────────────────────────────────────────────────────────

/// Load a CJK fallback font from the Windows system fonts directory.
fn setup_cjk_font(ctx: &egui::Context) {
    let mut fonts = egui::FontDefinitions::default();

    let candidates: &[(&str, &str)] = &[
        (r"C:\Windows\Fonts\msjh.ttc", "msjh"), // Microsoft JhengHei (TC)
        (r"C:\Windows\Fonts\msjhl.ttc", "msjhl"), // Microsoft JhengHei Light
        (r"C:\Windows\Fonts\msjhbd.ttc", "msjhbd"), // Microsoft JhengHei Bold
        (r"C:\Windows\Fonts\simsun.ttc", "simsun"), // SimSun (SC)
        (r"C:\Windows\Fonts\msyh.ttc", "msyh"), // Microsoft YaHei (SC)
        (r"C:\Windows\Fonts\yahei.ttc", "yahei"), // Microsoft YaHei alt
    ];

    for (path, name) in candidates {
        if let Ok(data) = std::fs::read(path) {
            fonts
                .font_data
                .insert(name.to_string(), egui::FontData::from_owned(data).into());
            // Push at *end* so Latin glyphs still use the default font.
            fonts
                .families
                .entry(egui::FontFamily::Proportional)
                .or_default()
                .push(name.to_string());
            fonts
                .families
                .entry(egui::FontFamily::Monospace)
                .or_default()
                .push(name.to_string());
            break;
        }
    }

    ctx.set_fonts(fonts);
}

// ── Entry point ────────────────────────────────────────────────────────────────

fn main() -> eframe::Result {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size(egui::vec2(850.0, 680.0))
            .with_min_inner_size(egui::vec2(600.0, 500.0)),
        ..Default::default()
    };

    eframe::run_native(
        Tr::new(Locale::ZhTw).app_title(),
        options,
        Box::new(|cc| {
            setup_cjk_font(&cc.egui_ctx);
            Ok(Box::new(TtsApp::default()))
        }),
    )
}
