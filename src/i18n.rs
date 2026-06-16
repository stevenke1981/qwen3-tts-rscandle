//! i18n — Lightweight localization for the GUI (zh-TW / en).
//!
//! All user-facing strings live here so translators only touch one file.

/// Supported UI locales.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Locale {
    /// Traditional Chinese (Taiwan) — default.
    ZhTw,
    /// English.
    En,
}

impl Locale {
    /// All available locales, their display labels, and short codes.
    pub fn variants() -> &'static [(Locale, &'static str, &'static str)] {
        &[
            (Locale::ZhTw, "中文 (繁體)", "zh-TW"),
            (Locale::En, "English", "en"),
        ]
    }

    /// Human-readable label for the current locale.
    pub fn label(&self) -> &'static str {
        match self {
            Locale::ZhTw => "中文 (繁體)",
            Locale::En => "English",
        }
    }
}

// ---------------------------------------------------------------------------
// Speaker names — proper nouns, kept as-is across locales
// ---------------------------------------------------------------------------

pub const SPEAKERS: &[&str] = &[
    "ryan", "serena", "vivian", "aiden", "uncle_fu", "ono_anna", "sohee", "eric", "dylan",
];

// ---------------------------------------------------------------------------
// Language options with localized display
// ---------------------------------------------------------------------------

pub struct LangOption {
    pub display: &'static str,
    pub internal: &'static str,
}

// ---------------------------------------------------------------------------
// Translator — one method per *unique* UI string
// ---------------------------------------------------------------------------

pub struct Tr {
    pub locale: Locale,
}

impl Tr {
    pub fn new(locale: Locale) -> Self {
        Self { locale }
    }

    // -- helpers -----------------------------------------------------------

    fn t<'a>(&self, zh: &'a str, en: &'a str) -> &'a str {
        match self.locale {
            Locale::ZhTw => zh,
            Locale::En => en,
        }
    }

    fn fmt(&self, zh: String, en: String) -> String {
        match self.locale {
            Locale::ZhTw => zh,
            Locale::En => en,
        }
    }

    // -- language list (localized) -----------------------------------------

    pub fn languages(&self) -> Vec<LangOption> {
        match self.locale {
            Locale::ZhTw => vec![
                LangOption {
                    display: "英文",
                    internal: "english",
                },
                LangOption {
                    display: "中文",
                    internal: "chinese",
                },
                LangOption {
                    display: "日文",
                    internal: "japanese",
                },
            ],
            Locale::En => vec![
                LangOption {
                    display: "English",
                    internal: "english",
                },
                LangOption {
                    display: "Chinese",
                    internal: "chinese",
                },
                LangOption {
                    display: "Japanese",
                    internal: "japanese",
                },
            ],
        }
    }

    // -- locale selector label --------------------------------------------

    pub fn ui_language_label(&self) -> &'static str {
        self.t("介面語言", "UI Language")
    }

    // -- app title ---------------------------------------------------------

    pub fn app_title(&self) -> &'static str {
        self.t("Qwen3-TTS 語音合成", "Qwen3-TTS")
    }

    // -- top bar -----------------------------------------------------------

    pub fn model_label(&self) -> &'static str {
        self.t("模型目錄：", "Model:")
    }

    pub fn output_label(&self) -> &'static str {
        self.t("輸出目錄：", "Output:")
    }

    // -- status bar --------------------------------------------------------

    pub fn loading_model(&self, secs: f64) -> String {
        self.fmt(
            format!("載入模型中… {secs:.1}s"),
            format!("Loading model… {secs:.1}s"),
        )
    }

    pub fn generating(&self, secs: f64) -> String {
        self.fmt(
            format!("生成中… {secs:.1}s"),
            format!("Generating… {secs:.1}s"),
        )
    }

    pub fn saving(&self) -> &'static str {
        self.t("儲存音頻中…", "Saving audio…")
    }

    pub fn done(&self, name: &str) -> String {
        self.fmt(format!("完成 — {name}"), format!("Done — {name}"))
    }

    pub fn error(&self, msg: &str) -> String {
        self.fmt(format!("錯誤：{msg}"), format!("Error: {msg}"))
    }

    pub fn files_generated(&self, count: usize) -> String {
        self.fmt(
            format!("已產生 {count} 個音檔"),
            format!("{count} audio file(s) generated"),
        )
    }

    // -- input section -----------------------------------------------------

    pub fn text_input_heading(&self) -> &'static str {
        self.t("輸入文字", "Text to Synthesize")
    }

    pub fn text_input_hint(&self) -> &'static str {
        self.t("請輸入要合成的文字…", "Enter text to synthesize…")
    }

    // -- batch -------------------------------------------------------------

    pub fn queue_batch(&self) -> &'static str {
        self.t("加入批次佇列", "Queue for batch")
    }

    pub fn queue_batch_count(&self, n: usize) -> String {
        self.fmt(
            format!("加入批次佇列（{n}）"),
            format!("Queue for batch ({n})"),
        )
    }

    pub fn clear_batch(&self) -> &'static str {
        self.t("清空批次", "Clear batch")
    }

    pub fn next_label(&self) -> &'static str {
        self.t("下一筆：", "Next:")
    }

    // -- parameters --------------------------------------------------------

    pub fn params_heading(&self) -> &'static str {
        self.t("參數", "Parameters")
    }

    pub fn speaker_label(&self) -> &'static str {
        self.t("說話者：", "Speaker:")
    }

    pub fn lang_label(&self) -> &'static str {
        self.t("合成語言：", "Language:")
    }

    pub fn duration_label(&self) -> &'static str {
        self.t("時長（秒）：", "Duration (s):")
    }

    pub fn seed_label(&self) -> &'static str {
        self.t("隨機種子：", "Seed:")
    }

    pub fn temperature_label(&self) -> &'static str {
        self.t("溫度：", "Temperature:")
    }

    pub fn topk_label(&self) -> &'static str {
        self.t("Top-K：", "Top-K:")
    }

    pub fn topp_label(&self) -> &'static str {
        self.t("Top-P：", "Top-P:")
    }

    pub fn rep_penalty_label(&self) -> &'static str {
        self.t("重複懲罰：", "Rep. Penalty:")
    }

    // -- generate button ---------------------------------------------------

    pub fn generating_text(&self) -> &'static str {
        self.t("生成中…", "Generating…")
    }

    pub fn generate_queued(&self, n: usize) -> String {
        self.fmt(
            format!("生成（佇列 {n}）"),
            format!("Generate ({n} queued)"),
        )
    }

    pub fn generate(&self) -> &'static str {
        self.t("生成", "Generate")
    }

    // -- inline progress ---------------------------------------------------

    pub fn loading_inline(&self, secs: f64) -> String {
        // same as status bar, could reuse
        self.loading_model(secs)
    }

    pub fn generating_inline(&self, secs: f64) -> String {
        self.generating(secs)
    }

    pub fn saving_inline(&self) -> &'static str {
        self.t("儲存中…", "Saving…")
    }

    // -- history -----------------------------------------------------------

    pub fn history_heading(&self) -> &'static str {
        self.t("已生成的音頻", "Generated Audio")
    }

    pub fn play_btn(&self) -> &'static str {
        self.t("播放", "Play")
    }

    pub fn open_btn(&self) -> &'static str {
        self.t("開啟資料夾", "Open")
    }
}
