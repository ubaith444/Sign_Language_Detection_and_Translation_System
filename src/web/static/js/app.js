document.addEventListener("DOMContentLoaded", () => {
  // DOM Elements - Status & Prediction
  const signNameEl = document.getElementById("detected-sign-name");
  const uncertaintyPillEl = document.getElementById("uncertainty-status-pill");
  const confidenceBarFillEl = document.getElementById("confidence-bar-fill");
  const confidenceValEl = document.getElementById("confidence-val");
  const telemetryFpsEl = document.getElementById("telemetry-fps");
  const activeModeTagEl = document.getElementById("active-mode-tag");
  const hudPrivacyBadge = document.getElementById("hud-privacy-badge");

  // DOM Elements - Sentence Builder
  const glossChipsContainer = document.getElementById("gloss-chips-container");
  const emptyGlossPlaceholder = document.getElementById("empty-gloss-placeholder");
  const translatedEnglishTextEl = document.getElementById("translated-english-text");
  const translationMethodLabelEl = document.getElementById("translation-method-label");
  const selectBuilderMode = document.getElementById("select-builder-mode");
  const btnAddWord = document.getElementById("btn-add-word");
  const btnAddSpace = document.getElementById("btn-add-space");
  const btnUndo = document.getElementById("btn-undo");
  const btnClear = document.getElementById("btn-clear");
  const btnSpeak = document.getElementById("btn-speak");
  const manualGlossInput = document.getElementById("manual-gloss-input");
  const btnSubmitManualGloss = document.getElementById("btn-submit-manual-gloss");

  // DOM Elements - Tabs
  const tabBtns = document.querySelectorAll(".tab-btn");
  const tabPanels = document.querySelectorAll(".tab-panel");

  // DOM Elements - Metrics & Matrix
  const metricAccuracy = document.getElementById("metric-accuracy");
  const metricMacroF1 = document.getElementById("metric-macro-f1");
  const metricWer = document.getElementById("metric-wer");
  const metricLatency = document.getElementById("metric-latency");
  const siTestSigner = document.getElementById("si-test-signer");
  const siAccuracy = document.getElementById("si-accuracy");
  const siGap = document.getElementById("si-gap");
  const matrixContainer = document.getElementById("matrix-container");
  const btnRunBenchmark = document.getElementById("btn-run-benchmark");

  // DOM Elements - Settings & Lexicon
  const selectDetectionMode = document.getElementById("select-detection-mode");
  const confidenceThresholdSlider = document.getElementById("confidence-threshold-slider");
  const thresholdValLabel = document.getElementById("threshold-val-label");
  const togglePrivacyBlur = document.getElementById("toggle-privacy-blur");
  const btnRetrainModels = document.getElementById("btn-retrain-models");
  const lexiconListContainer = document.getElementById("lexicon-list-container");
  const btnShowAlphabet = document.getElementById("btn-show-alphabet");
  const btnShowWords = document.getElementById("btn-show-words");

  // State
  let currentPrediction = {
    sign: "Waiting...",
    confidence: 0.0,
    is_uncertain: true
  };
  let cachedLexicon = {};
  let currentLexiconView = "alphabet";

  // 1. Tab Switching Logic
  tabBtns.forEach(btn => {
    btn.addEventListener("click", () => {
      tabBtns.forEach(b => b.classList.remove("active"));
      tabPanels.forEach(p => p.classList.remove("active"));

      btn.classList.add("active");
      const targetPanel = document.getElementById(btn.getAttribute("data-tab"));
      if (targetPanel) {
        targetPanel.classList.add("active");
      }
    });
  });

  const urlParams = new URLSearchParams(window.location.search);
  const targetTab = urlParams.get("tab") || window.location.hash.replace("#", "");
  if (targetTab) {
    const targetBtn = document.querySelector(`.tab-btn[data-tab="${targetTab}"]`);
    if (targetBtn) {
      targetBtn.click();
    }
  }

  // 2. Telemetry & State Polling
  async function fetchLiveState() {
    try {
      const response = await fetch("/api/state");
      if (!response.ok) return;
      const data = await response.json();

      // Update Prediction Banner
      currentPrediction.sign = data.current_sign || "UNKNOWN";
      currentPrediction.confidence = data.confidence || 0.0;
      currentPrediction.is_uncertain = !!data.is_uncertain;

      signNameEl.textContent = currentPrediction.sign;
      const pct = Math.round(currentPrediction.confidence * 100);
      confidenceBarFillEl.style.width = `${pct}%`;
      confidenceValEl.textContent = `${pct}%`;

      if (currentPrediction.is_uncertain || currentPrediction.sign.includes("UNCERTAIN") || currentPrediction.sign === "UNKNOWN / UNCERTAIN") {
        uncertaintyPillEl.textContent = "Uncertain";
        uncertaintyPillEl.className = "uncertainty-pill uncertain";
      } else {
        uncertaintyPillEl.textContent = "Calibrated";
        uncertaintyPillEl.className = "uncertainty-pill known";
      }

      // Update Telemetry
      if (data.telemetry) {
        telemetryFpsEl.textContent = `FPS: ${data.telemetry.effective_fps || 0} | Latency: ${data.telemetry.pipeline_latency_ms || 0}ms`;
      }

      // Update Facial Modality HUD
      if (data.facial_modality) {
        const modPill = document.getElementById("hud-modality-pill");
        const modCue = document.getElementById("hud-modality-cue");
        if (modPill) {
          modPill.textContent = data.facial_modality.modality_label || "Declarative Statement";
          const mod = data.facial_modality.modality || "";
          if (mod.includes("QUESTION")) {
            modPill.style.background = "#eff6ff";
            modPill.style.color = "#1d4ed8";
          } else if (mod.includes("NEGATIVE")) {
            modPill.style.background = "#fff7ed";
            modPill.style.color = "#c2410c";
          } else {
            modPill.style.background = "#f0fdf4";
            modPill.style.color = "#15803d";
          }
        }
        if (modCue) {
          modCue.textContent = (data.facial_modality.cues || []).join(", ") || "";
        }
      }

      // Update HamNoSys Phonology HUD
      if (data.phonology) {
        const hamSummary = document.getElementById("hud-hamnosys-summary");
        const hamCode = document.getElementById("hud-hamnosys-code");
        if (hamSummary) {
          hamSummary.textContent = data.phonology.phonological_summary || "--";
        }
        if (hamCode) {
          hamCode.textContent = data.phonology.hamnosys_transcription || "--";
        }
      }

      // Update Sentence Builder View
      renderSentenceState(data.sentence_state);
    } catch (err) {
      // Stream polling silent catch
    }
  }

  // 3. Render Sentence Chips and Translation Text
  function renderSentenceState(state) {
    if (!state) return;

    const tokens = state.tokens || state.glosses || [];
    glossChipsContainer.innerHTML = "";

    if (tokens.length === 0) {
      glossChipsContainer.appendChild(emptyGlossPlaceholder);
    } else {
      tokens.forEach((t) => {
        const chip = document.createElement("span");
        chip.className = "gloss-chip";
        chip.textContent = t === " " ? "[SPACE]" : t;
        if (t === " ") {
          chip.style.backgroundColor = "#f1f5f9";
          chip.style.borderColor = "#cbd5e1";
          chip.style.color = "#475569";
        }
        glossChipsContainer.appendChild(chip);
      });
    }

    if (state.translated_text) {
      translatedEnglishTextEl.textContent = state.translated_text;
      const method = state.translation_method === "exact_corpus_match" 
        ? "Verified Corpus Match" 
        : (state.translation_method === "fingerspelling_assembly" ? "Continuous Fingerspelling Assembly" : "Syntactic Grammar Transformer");
      translationMethodLabelEl.textContent = `Syntactic Output (${method})`;
    } else {
      translatedEnglishTextEl.textContent = "Awaiting sign input for translation...";
      translationMethodLabelEl.textContent = "Syntactic English Translation";
    }
  }

  // 4. Action Handlers (Add, Space, Delete, Clear, Speak)
  btnAddWord.addEventListener("click", async () => {
    if (currentPrediction.sign && !currentPrediction.sign.includes("UNCERTAIN") && currentPrediction.sign !== "UNKNOWN / UNCERTAIN") {
      try {
        const res = await fetch("/api/sentence/add", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ sign: currentPrediction.sign, confidence: currentPrediction.confidence })
        });
        const data = await res.json();
        renderSentenceState(data);
      } catch (e) {
        console.error("Add word failed:", e);
      }
    }
  });

  btnAddSpace.addEventListener("click", async () => {
    try {
      const res = await fetch("/api/sentence/space", { method: "POST" });
      const data = await res.json();
      renderSentenceState(data);
    } catch (e) {
      console.error("Space insert failed:", e);
    }
  });

  btnUndo.addEventListener("click", async () => {
    try {
      const res = await fetch("/api/sentence/delete", { method: "POST" });
      const data = await res.json();
      renderSentenceState(data);
    } catch (e) {
      console.error("Delete failed:", e);
    }
  });

  btnClear.addEventListener("click", async () => {
    try {
      const res = await fetch("/api/sentence/clear", { method: "POST" });
      const data = await res.json();
      renderSentenceState(data);
    } catch (e) {
      console.error("Clear failed:", e);
    }
  });

  btnSpeak.addEventListener("click", async () => {
    const textToSpeak = translatedEnglishTextEl.textContent;
    if (!textToSpeak || textToSpeak.startsWith("Awaiting")) return;

    fetch("/api/sentence/speak", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: textToSpeak })
    }).catch(() => {});

    if ("speechSynthesis" in window) {
      window.speechSynthesis.cancel();
      const utterance = new SpeechSynthesisUtterance(textToSpeak);
      utterance.rate = 1.0;
      utterance.pitch = 1.0;
      window.speechSynthesis.speak(utterance);
    }
  });

  selectBuilderMode.addEventListener("change", async (e) => {
    const mode = e.target.value;
    try {
      const res = await fetch("/api/sentence/builder_mode", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ mode })
      });
      const data = await res.json();
      renderSentenceState(data);
    } catch (err) {
      console.error("Mode switch failed:", err);
    }
  });

  btnSubmitManualGloss.addEventListener("click", async () => {
    const text = manualGlossInput.value.trim();
    if (!text) return;
    try {
      const res = await fetch("/api/sentence/add", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sign: text, confidence: 1.0 })
      });
      const data = await res.json();
      renderSentenceState(data);
      manualGlossInput.value = "";
    } catch (e) {
      console.error("Manual gloss insertion failed:", e);
    }
  });

  // 5. Settings Handlers
  selectDetectionMode.addEventListener("change", async (e) => {
    const mode = e.target.value;
    activeModeTagEl.textContent = `Mode: ${mode === "STATIC" ? "Static Manual Signs & Alphabet" : "Dynamic Movement Sequences"}`;
    await fetch("/api/settings/mode", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode })
    });
  });

  confidenceThresholdSlider.addEventListener("input", (e) => {
    const val = e.target.value;
    thresholdValLabel.textContent = `${val}%`;
  });

  confidenceThresholdSlider.addEventListener("change", async (e) => {
    const threshold = parseFloat(e.target.value) / 100.0;
    await fetch("/api/settings/threshold", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ threshold })
    });
  });

  togglePrivacyBlur.addEventListener("change", async (e) => {
    const enabled = e.target.checked;
    hudPrivacyBadge.textContent = enabled ? "Privacy: Blur Active" : "Privacy: Standard";
    await fetch("/api/settings/privacy", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ privacy_blur: enabled })
    });
  });

  btnRetrainModels.addEventListener("click", async () => {
    btnRetrainModels.disabled = true;
    btnRetrainModels.textContent = "Retraining Models (Generating Prototypes)...";
    try {
      const res = await fetch("/api/models/retrain", { method: "POST" });
      const result = await res.json();
      alert("Model retraining complete! Acc: " + (result.static_train_acc * 100).toFixed(1) + "%");
      fetchEvaluationMetrics();
    } catch (e) {
      alert("Retraining error: " + e.message);
    } finally {
      btnRetrainModels.disabled = false;
      btnRetrainModels.textContent = "Retrain Models with Synthetic Prototypes";
    }
  });

  // 5.1 Real-Time Live Webcam Fine-Tuning & Calibration
  const selectFinetuneSign = document.getElementById("select-finetune-sign");
  const btnCaptureFinetuneSample = document.getElementById("btn-capture-finetune-sample");
  const btnRunFinetune = document.getElementById("btn-run-finetune");
  const btnResetFinetune = document.getElementById("btn-reset-finetune");
  const finetuneSampleBadge = document.getElementById("finetune-sample-badge");
  const finetuneFeedbackBox = document.getElementById("finetune-feedback-box");
  const finetuneFeedbackText = document.getElementById("finetune-feedback-text");
  const inputDatasetFolder = document.getElementById("input-dataset-folder");
  const btnImportDataset = document.getElementById("btn-import-dataset");
  const datasetImportStatus = document.getElementById("dataset-import-status");

  async function updateFinetuneStatus() {
    try {
      const res = await fetch("/api/finetune/status");
      if (!res.ok) return;
      const data = await res.json();
      const selected = selectFinetuneSign ? selectFinetuneSign.value : "A";
      const countForSelected = (data.sample_counts && data.sample_counts[selected]) || 0;
      if (finetuneSampleBadge) {
        finetuneSampleBadge.textContent = `${data.total_user_samples} Total (${countForSelected} for '${selected}')`;
      }
    } catch (e) {
      console.warn("Could not fetch finetune status:", e);
    }
  }

  if (selectFinetuneSign) {
    selectFinetuneSign.addEventListener("change", updateFinetuneStatus);
  }

  if (btnCaptureFinetuneSample) {
    btnCaptureFinetuneSample.addEventListener("click", async () => {
      const label = selectFinetuneSign.value;
      btnCaptureFinetuneSample.disabled = true;
      btnCaptureFinetuneSample.textContent = "Capturing...";
      try {
        const res = await fetch("/api/finetune/capture_sample", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ label })
        });
        const data = await res.json();
        if (res.ok) {
          finetuneFeedbackBox.style.display = "block";
          finetuneFeedbackText.textContent = `Captured sample #${data.sample_count} for sign '${data.label}'. (Total pool: ${data.total_user_samples})`;
          finetuneFeedbackBox.style.borderColor = "#bbf7d0";
          finetuneFeedbackBox.style.backgroundColor = "#f0fdf4";
          updateFinetuneStatus();
        } else {
          finetuneFeedbackBox.style.display = "block";
          finetuneFeedbackText.textContent = data.error || "Capture failed. Ensure hands are clearly visible in camera.";
          finetuneFeedbackBox.style.borderColor = "#fecaca";
          finetuneFeedbackBox.style.backgroundColor = "#fef2f2";
        }
      } catch (err) {
        finetuneFeedbackBox.style.display = "block";
        finetuneFeedbackText.textContent = "Network error during capture.";
      } finally {
        btnCaptureFinetuneSample.disabled = false;
        btnCaptureFinetuneSample.textContent = "Capture Frame";
      }
    });
  }

  if (btnRunFinetune) {
    btnRunFinetune.addEventListener("click", async () => {
      btnRunFinetune.disabled = true;
      btnRunFinetune.textContent = "Fine-Tuning Active Model (< 2s)...";
      try {
        const res = await fetch("/api/finetune/train", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ model_type: "rf" })
        });
        const data = await res.json();
        if (res.ok) {
          finetuneFeedbackBox.style.display = "block";
          finetuneFeedbackText.innerHTML = `<strong>Fine-Tuning Success!</strong> Retrained in <strong>${data.retraining_time_sec}s</strong>. User Calibration Accuracy: <strong>${(data.user_calibration_accuracy * 100).toFixed(1)}%</strong> on ${data.user_raw_samples} raw samples (${data.user_augmented_samples} augmented).`;
          finetuneFeedbackBox.style.borderColor = "#bbf7d0";
          finetuneFeedbackBox.style.backgroundColor = "#f0fdf4";
          fetchEvaluationMetrics();
        } else {
          finetuneFeedbackBox.style.display = "block";
          finetuneFeedbackText.textContent = data.error || "Fine-tuning failed. Capture samples first.";
          finetuneFeedbackBox.style.borderColor = "#fecaca";
          finetuneFeedbackBox.style.backgroundColor = "#fef2f2";
        }
      } catch (err) {
        finetuneFeedbackBox.style.display = "block";
        finetuneFeedbackText.textContent = "Error executing fine-tuning: " + err.message;
      } finally {
        btnRunFinetune.disabled = false;
        btnRunFinetune.textContent = "Fine-Tune Model Now (1-Click)";
      }
    });
  }

  if (btnResetFinetune) {
    btnResetFinetune.addEventListener("click", async () => {
      btnResetFinetune.disabled = true;
      try {
        await fetch("/api/finetune/reset", { method: "POST" });
        finetuneFeedbackBox.style.display = "block";
        finetuneFeedbackText.textContent = "Reset calibration to factory baseline models.";
        updateFinetuneStatus();
        fetchEvaluationMetrics();
      } catch (e) {
        console.error("Reset failed:", e);
      } finally {
        btnResetFinetune.disabled = false;
      }
    });
  }

  if (btnImportDataset) {
    btnImportDataset.addEventListener("click", async () => {
      const path = inputDatasetFolder.value.trim();
      if (!path) {
        alert("Please enter a valid directory path containing alphabet images (e.g. D:/Sign Language/data/isl_alphabet_dataset)");
        return;
      }
      btnImportDataset.disabled = true;
      btnImportDataset.textContent = "Importing & Extracting...";
      datasetImportStatus.style.display = "block";
      datasetImportStatus.textContent = "Scanning directory and extracting MediaPipe landmarks...";

      try {
        const res = await fetch("/api/dataset/import_directory", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ input_dir: path, auto_train: true })
        });
        const data = await res.json();
        if (res.ok && data.status === "success") {
          datasetImportStatus.innerHTML = `<span style="color: var(--color-success); font-weight: 600;">Imported ${data.total_classes} classes (${data.total_landmarks_extracted} frames) in ${data.extraction_time_sec}s!</span>`;
          fetchEvaluationMetrics();
        } else {
          datasetImportStatus.innerHTML = `<span style="color: var(--color-error);">${data.error || data.message || "Import failed"}</span>`;
        }
      } catch (err) {
        datasetImportStatus.innerHTML = `<span style="color: var(--color-error);">Network error: ${err.message}</span>`;
      } finally {
        btnImportDataset.disabled = false;
        btnImportDataset.textContent = "Import & Train";
      }
    });
  }

  updateFinetuneStatus();

  // 6. Evaluation Dashboard & Confusion Matrix
  async function fetchEvaluationMetrics() {
    try {
      const res = await fetch("/api/benchmark");
      if (res.ok) {
        const data = await res.json();
        if (data.static_classification) {
          metricAccuracy.textContent = `${(data.static_classification.accuracy * 100).toFixed(1)}%`;
          metricMacroF1.textContent = `${(data.static_classification.macro_f1 * 100).toFixed(1)}%`;
        }
        if (data.translation_evaluation) {
          metricWer.textContent = `${(data.translation_evaluation.average_wer * 100).toFixed(1)}%`;
        }
        if (data.performance_profile) {
          metricLatency.textContent = `${data.performance_profile.average_inference_latency_ms}ms`;
        }
        renderConfusionMatrix(data.static_classification);
      }

      // Fetch Signer-Independent Disjoint Split Results
      const siRes = await fetch("/api/signer_independent");
      if (siRes.ok) {
        const siData = await siRes.json();
        siTestSigner.textContent = siData.held_out_test_signer || "signer_04";
        const siAcc = siData.signer_independent_metrics ? siData.signer_independent_metrics.accuracy : 1.0;
        siAccuracy.textContent = `${(siAcc * 100).toFixed(1)}%`;
        siGap.textContent = `${((siData.generalization_gap || 0.0) * 100).toFixed(1)}%`;
      }
    } catch (e) {
      console.error("Evaluation metrics load failed:", e);
    }
  }

  function renderConfusionMatrix(classData) {
    if (!classData || !classData.confusion_matrix || !classData.classes) return;

    const classes = classData.classes.slice(0, 10);
    const cm = classData.confusion_matrix;

    let html = '<table class="matrix-table"><thead><tr><th>Pred &rarr;<br>True &darr;</th>';
    classes.forEach(c => {
      html += `<th>${c.substring(0, 4)}</th>`;
    });
    html += '</tr></thead><tbody>';

    classes.forEach((rowClass, r) => {
      html += `<tr><th>${rowClass.substring(0, 4)}</th>`;
      classes.forEach((colClass, c) => {
        const val = cm[r] ? cm[r][c] : 0;
        const isDiag = r === c;
        html += `<td class="${isDiag ? 'matrix-cell-diag' : ''}">${val}</td>`;
      });
      html += '</tr>';
    });
    html += '</tbody></table>';

    matrixContainer.innerHTML = html;
  }

  btnRunBenchmark.addEventListener("click", async () => {
    btnRunBenchmark.disabled = true;
    btnRunBenchmark.textContent = "Executing Automated Benchmark Suite...";
    try {
      const res = await fetch("/api/benchmark/run", { method: "POST" });
      await res.json();
      fetchEvaluationMetrics();
    } catch (e) {
      console.error("Benchmark failed:", e);
    } finally {
      btnRunBenchmark.disabled = false;
      btnRunBenchmark.textContent = "Run Quantitative Benchmark Suite";
    }
  });

  // 7. Load ISL Lexicon & Alphabet
  async function loadLexicon() {
    try {
      const res = await fetch("/api/lexicon");
      if (!res.ok) return;
      cachedLexicon = await res.json();
      renderLexiconView();
    } catch (e) {
      lexiconListContainer.innerHTML = "<p>Unable to load lexicon definitions.</p>";
    }
  }

  function renderLexiconView() {
    let html = "";
    if (currentLexiconView === "alphabet") {
      const alphabet = (cachedLexicon.alphabet_dataset_a && cachedLexicon.alphabet_dataset_a.alphabet) || {};
      for (const [letter, info] of Object.entries(alphabet)) {
        html += `
          <div class="lexicon-item">
            <div class="lexicon-item-header">
              <span class="lexicon-term">Letter ${letter}</span>
              <span class="lexicon-category">Dataset A (${info.type})</span>
            </div>
            <p class="lexicon-desc">${info.description} <em>[${info.category}]</em></p>
          </div>
        `;
      }
    } else {
      const staticSigns = cachedLexicon.static_signs || {};
      const dynamicSigns = cachedLexicon.dynamic_signs || {};

      for (const [key, val] of Object.entries(staticSigns)) {
        html += `
          <div class="lexicon-item">
            <div class="lexicon-item-header">
              <span class="lexicon-term">${key}</span>
              <span class="lexicon-category">Static (${val.type})</span>
            </div>
            <p class="lexicon-desc">${val.description} <em>Meaning: ${val.english_meaning}</em></p>
          </div>
        `;
      }
      for (const [key, val] of Object.entries(dynamicSigns)) {
        html += `
          <div class="lexicon-item">
            <div class="lexicon-item-header">
              <span class="lexicon-term">${key}</span>
              <span class="lexicon-category">Dynamic Movement</span>
            </div>
            <p class="lexicon-desc">${val.description} <em>Meaning: ${val.english_meaning}</em></p>
          </div>
        `;
      }
    }
    lexiconListContainer.innerHTML = html;
  }

  btnShowAlphabet.addEventListener("click", () => {
    currentLexiconView = "alphabet";
    btnShowAlphabet.className = "btn btn-primary";
    btnShowWords.className = "btn btn-secondary";
    renderLexiconView();
  });

  btnShowWords.addEventListener("click", () => {
    currentLexiconView = "words";
    btnShowWords.className = "btn btn-primary";
    btnShowAlphabet.className = "btn btn-secondary";
    renderLexiconView();
  });

  // 8. Video Batch Processing & Subtitle Exporter
  const batchVideoInput = document.getElementById("batch-video-input");
  const dropzoneArea = document.getElementById("dropzone-area");
  const selectedFileLabel = document.getElementById("selected-file-label");
  const batchModeSelect = document.getElementById("batch-mode-select");
  const batchBurnSubsCheckbox = document.getElementById("batch-burn-subs-checkbox");
  const btnProcessUploadedVideo = document.getElementById("btn-process-uploaded-video");
  const btnProcessSampleVideo = document.getElementById("btn-process-sample-video");
  const batchStatusBox = document.getElementById("batch-status-box");
  const batchStatusText = document.getElementById("batch-status-text");
  const batchResultsContainer = document.getElementById("batch-results-container");
  const batchStatDuration = document.getElementById("batch-stat-duration");
  const batchStatFrames = document.getElementById("batch-stat-frames");
  const batchStatFps = document.getElementById("batch-stat-fps");
  const batchStatSegments = document.getElementById("batch-stat-segments");
  const batchTranslatedText = document.getElementById("batch-translated-text");
  const batchGlossesList = document.getElementById("batch-glosses-list");
  const btnDlSrt = document.getElementById("btn-dl-srt");
  const btnDlVtt = document.getElementById("btn-dl-vtt");
  const btnDlJson = document.getElementById("btn-dl-json");
  const btnDlMp4 = document.getElementById("btn-dl-mp4");
  const batchSegmentsTbody = document.getElementById("batch-segments-tbody");

  let selectedBatchFile = null;

  if (dropzoneArea && batchVideoInput) {
    dropzoneArea.addEventListener("click", () => batchVideoInput.click());

    batchVideoInput.addEventListener("change", (e) => {
      if (e.target.files && e.target.files[0]) {
        handleBatchFileSelect(e.target.files[0]);
      }
    });

    dropzoneArea.addEventListener("dragover", (e) => {
      e.preventDefault();
      dropzoneArea.classList.add("dragover");
    });

    dropzoneArea.addEventListener("dragleave", () => {
      dropzoneArea.classList.remove("dragover");
    });

    dropzoneArea.addEventListener("drop", (e) => {
      e.preventDefault();
      dropzoneArea.classList.remove("dragover");
      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        handleBatchFileSelect(e.dataTransfer.files[0]);
      }
    });
  }

  function handleBatchFileSelect(file) {
    selectedBatchFile = file;
    selectedFileLabel.style.display = "block";
    selectedFileLabel.textContent = `Selected: ${file.name} (${(file.size / (1024 * 1024)).toFixed(2)} MB)`;
    btnProcessUploadedVideo.disabled = false;
  }

  async function executeBatchVideoProcess(file) {
    const isSample = !file;
    batchStatusBox.style.display = "flex";
    batchStatusText.textContent = isSample 
      ? "Processing benchmark clip through holistic model pipeline..." 
      : `Ingesting and analyzing ${file.name}...`;
    batchResultsContainer.style.display = "none";
    btnProcessUploadedVideo.disabled = true;
    btnProcessSampleVideo.disabled = true;

    const formData = new FormData();
    if (file) {
      formData.append("video_file", file);
    }
    formData.append("burn_subtitles", batchBurnSubsCheckbox.checked ? "true" : "false");
    formData.append("mode", batchModeSelect.value);

    try {
      const response = await fetch("/api/video/process", {
        method: "POST",
        body: formData
      });

      if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.error || `Server responded with status ${response.status}`);
      }

      const result = await response.json();
      renderBatchResults(result);
    } catch (err) {
      alert("Batch processing error: " + err.message);
    } finally {
      batchStatusBox.style.display = "none";
      btnProcessUploadedVideo.disabled = !selectedBatchFile;
      btnProcessSampleVideo.disabled = false;
    }
  }

  function renderBatchResults(data) {
    batchResultsContainer.style.display = "flex";

    // Telemetry
    const duration = data.duration_sec ?? data.duration_seconds ?? 0;
    batchStatDuration.textContent = `${duration.toFixed(1)}s`;
    batchStatFrames.textContent = data.total_frames || 0;
    batchStatFps.textContent = `${(data.processing_fps || 0).toFixed(1)} FPS`;

    const segments = data.aligned_sign_segments || data.sign_segments || [];
    batchStatSegments.textContent = segments.length;

    // Translation Output
    const translatedText = data.translated_english ?? data.syntactic_english_translation ?? "(No sign sequence generated)";
    batchTranslatedText.textContent = translatedText;

    const glosses = data.recognized_gloss_tokens ?? data.gloss_sequence ?? [];
    batchGlossesList.textContent = (glosses && glosses.length > 0)
      ? glosses.join(", ")
      : "(None)";

    // Subtitle Download URLs
    const vName = encodeURIComponent(data.video_name);
    btnDlSrt.href = `/api/video/download/srt/${vName}`;
    btnDlVtt.href = `/api/video/download/vtt/${vName}`;
    btnDlJson.href = `/api/video/download/json/${vName}`;

    if (data.exported_files && (data.exported_files.burned_video || data.exported_files.subtitled_video_mp4)) {
      btnDlMp4.href = `/api/video/download/mp4/${vName}`;
      btnDlMp4.style.display = "inline-flex";
    } else {
      btnDlMp4.style.display = "none";
    }

    // Segments Table
    if (segments.length === 0) {
      batchSegmentsTbody.innerHTML = `<tr><td colspan="5" style="text-align:center; color: var(--color-text-muted);">No distinct signs detected in video segment.</td></tr>`;
    } else {
      let rowsHtml = "";
      segments.forEach((seg, idx) => {
        const confPct = Math.round((seg.confidence || 0) * 100);
        const startTime = seg.start_time_sec ?? seg.start_time_seconds ?? 0;
        const endTime = seg.end_time_sec ?? seg.end_time_seconds ?? 0;
        const dur = (seg.duration_sec ?? seg.duration_seconds ?? Math.max(0, endTime - startTime)).toFixed(2);
        const srtStart = seg.start_timestamp || seg.start_timestamp_srt || "--";
        const srtEnd = seg.end_timestamp || seg.end_timestamp_srt || "--";
        rowsHtml += `
          <tr>
            <td>${seg.cue_index || idx + 1}</td>
            <td style="font-family: var(--font-family-mono); font-size: 0.6875rem;">${srtStart} &rarr; ${srtEnd}</td>
            <td>${dur}s</td>
            <td><strong style="color: var(--color-primary);">${seg.sign}</strong></td>
            <td>${confPct}%</td>
          </tr>
        `;
      });
      batchSegmentsTbody.innerHTML = rowsHtml;
    }
  }

  if (btnProcessUploadedVideo) {
    btnProcessUploadedVideo.addEventListener("click", () => {
      if (selectedBatchFile) {
        executeBatchVideoProcess(selectedBatchFile);
      }
    });
  }

  if (btnProcessSampleVideo) {
    btnProcessSampleVideo.addEventListener("click", () => {
      executeBatchVideoProcess(null);
    });
  }

  // Automatic sample runner for automated testing
  if (urlParams.get("auto_sample") === "true") {
    setTimeout(() => {
      executeBatchVideoProcess(null);
    }, 800);
  }

  // 9. Two-Way Communication & Canvas Sign Player
  const twowaySpeechInput = document.getElementById("twoway-speech-input");
  const btnListenMic = document.getElementById("btn-listen-mic");
  const btnTranslatePlayIsl = document.getElementById("btn-translate-play-isl");
  const playerGlossSequence = document.getElementById("player-gloss-sequence");
  const playerModalityBadge = document.getElementById("player-modality-badge");
  const signPlayerCanvas = document.getElementById("sign-player-canvas");
  const playerStatusBadge = document.getElementById("player-status-badge");
  const playerSignCue = document.getElementById("player-sign-cue");
  const playerHamnosysCue = document.getElementById("player-hamnosys-cue");
  const playerTimelineCue = document.getElementById("player-timeline-cue");
  const btnPlayerStepPrev = document.getElementById("btn-player-step-prev");
  const btnPlayerPlay = document.getElementById("btn-player-play");
  const btnPlayerPause = document.getElementById("btn-player-pause");
  const btnPlayerStepNext = document.getElementById("btn-player-step-next");
  const playerSpeedSelect = document.getElementById("player-speed-select");

  let playerQueue = [];
  let currentSignIdx = 0;
  let currentFrameIdx = 0;
  let isPlaying = false;
  let playerTimer = null;

  // Speech Recognition (Microphone)
  let speechRec = null;
  if ("webkitSpeechRecognition" in window || "SpeechRecognition" in window) {
    const SpeechAPI = window.SpeechRecognition || window.webkitSpeechRecognition;
    speechRec = new SpeechAPI();
    speechRec.continuous = false;
    speechRec.interimResults = false;
    speechRec.lang = "en-US";

    speechRec.onresult = (e) => {
      const text = e.results[0][0].transcript;
      if (twowaySpeechInput) twowaySpeechInput.value = text;
      if (btnListenMic) {
        btnListenMic.textContent = "Listen Microphone (Voice)";
        btnListenMic.className = "btn btn-primary";
      }
      executeTwoWayTranslate(text);
    };

    speechRec.onerror = () => {
      if (btnListenMic) {
        btnListenMic.textContent = "Listen Microphone (Voice)";
        btnListenMic.className = "btn btn-primary";
      }
    };

    speechRec.onend = () => {
      if (btnListenMic) {
        btnListenMic.textContent = "Listen Microphone (Voice)";
        btnListenMic.className = "btn btn-primary";
      }
    };
  }

  if (btnListenMic) {
    btnListenMic.addEventListener("click", () => {
      if (!speechRec) {
        alert("Speech Recognition is not supported in this browser. Please type English in the input box.");
        return;
      }
      try {
        speechRec.start();
        btnListenMic.textContent = "Listening to voice...";
        btnListenMic.className = "btn btn-destructive";
      } catch (err) {
        speechRec.stop();
        btnListenMic.textContent = "Listen Microphone (Voice)";
        btnListenMic.className = "btn btn-primary";
      }
    });
  }

  if (btnTranslatePlayIsl) {
    btnTranslatePlayIsl.addEventListener("click", () => {
      const text = (twowaySpeechInput ? twowaySpeechInput.value : "").trim();
      if (!text) return;
      executeTwoWayTranslate(text);
    });
  }

  async function executeTwoWayTranslate(text) {
    try {
      if (playerStatusBadge) playerStatusBadge.textContent = "Translating...";
      const res = await fetch("/api/two_way/translate_english", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text })
      });
      const data = await res.json();

      if (playerGlossSequence) playerGlossSequence.textContent = data.gloss_string || "No signs generated";
      if (playerModalityBadge) {
        playerModalityBadge.textContent = data.modality.replace("_", " ");
      }

      // Load animation frames for each sign
      playerQueue = [];
      const signs = data.isl_gloss_sequence || [];
      for (const s of signs) {
        try {
          const aRes = await fetch(`/api/two_way/sign_animation/${encodeURIComponent(s)}`);
          if (aRes.ok) {
            const anim = await aRes.json();
            playerQueue.push(anim);
          }
        } catch (e) {}
      }

      if (playerQueue.length > 0) {
        currentSignIdx = 0;
        currentFrameIdx = 0;
        startPlayerPlayback();
      } else {
        if (playerStatusBadge) playerStatusBadge.textContent = "No sign animation found";
      }
      fetchSessionHistory();
    } catch (err) {
      console.error("Two-way translation failed:", err);
    }
  }

  function startPlayerPlayback() {
    isPlaying = true;
    if (playerStatusBadge) playerStatusBadge.textContent = "Playing ISL Signs";
    if (playerTimer) clearInterval(playerTimer);

    const speed = parseFloat(playerSpeedSelect ? playerSpeedSelect.value : "1.0");
    const intervalMs = Math.round(450 / speed);

    playerTimer = setInterval(() => {
      if (!isPlaying || playerQueue.length === 0) return;

      const currentAnim = playerQueue[currentSignIdx];
      if (!currentAnim || !currentAnim.frames || currentAnim.frames.length === 0) {
        advanceNextSign();
        return;
      }

      const frame = currentAnim.frames[currentFrameIdx];
      drawSignSkeleton(frame, currentAnim.sign, currentAnim.hamnosys);

      currentFrameIdx++;
      if (currentFrameIdx >= currentAnim.frames.length) {
        currentFrameIdx = 0;
        advanceNextSign();
      }
    }, intervalMs);
  }

  function advanceNextSign() {
    currentSignIdx++;
    if (currentSignIdx >= playerQueue.length) {
      currentSignIdx = playerQueue.length - 1;
      isPlaying = false;
      if (playerTimer) clearInterval(playerTimer);
      if (playerStatusBadge) playerStatusBadge.textContent = "Sequence Completed";
    }
  }

  function drawSignSkeleton(frame, signName, hamnosys) {
    if (!signPlayerCanvas) return;
    const ctx = signPlayerCanvas.getContext("2d");
    const w = signPlayerCanvas.width;
    const h = signPlayerCanvas.height;

    ctx.clearRect(0, 0, w, h);

    // Background
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, w, h);

    // Draw grid guide
    ctx.strokeStyle = "#f1f5f9";
    ctx.lineWidth = 1;
    for (let x = 40; x < w; x += 40) {
      ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, h); ctx.stroke();
    }
    for (let y = 40; y < h; y += 40) {
      ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
    }

    if (!frame) return;

    const toX = (val) => val * w;
    const toY = (val) => val * h;

    const head = frame.head || [0.5, 0.2];
    const ls = frame.left_shoulder || [0.35, 0.35];
    const rs = frame.right_shoulder || [0.65, 0.35];
    const le = frame.left_elbow || [ls[0] - 0.05, ls[1] + 0.15];
    const re = frame.right_elbow || [rs[0] + 0.05, rs[1] + 0.15];
    const lw = frame.left_wrist || [0.30, 0.65];
    const rw = frame.right_wrist || [0.55, 0.45];

    // Spine and Shoulders
    ctx.lineWidth = 4;
    ctx.strokeStyle = "#64748b";
    ctx.lineCap = "round";

    // Shoulder bridge
    ctx.beginPath();
    ctx.moveTo(toX(ls[0]), toY(ls[1]));
    ctx.lineTo(toX(rs[0]), toY(rs[1]));
    ctx.stroke();

    // Neck to Head
    const midShoulder = [(ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2];
    ctx.beginPath();
    ctx.moveTo(toX(midShoulder[0]), toY(midShoulder[1]));
    ctx.lineTo(toX(head[0]), toY(head[1] + 0.06));
    ctx.stroke();

    // Torso Spine
    ctx.beginPath();
    ctx.moveTo(toX(midShoulder[0]), toY(midShoulder[1]));
    ctx.lineTo(toX(midShoulder[0]), toY(0.75));
    ctx.stroke();

    // Left Arm (Viewer Right)
    ctx.strokeStyle = "#0284c7";
    ctx.beginPath();
    ctx.moveTo(toX(ls[0]), toY(ls[1]));
    ctx.lineTo(toX(le[0]), toY(le[1]));
    ctx.lineTo(toX(lw[0]), toY(lw[1]));
    ctx.stroke();

    // Right Arm (Dominant Signer Arm)
    ctx.strokeStyle = "#1d4ed8";
    ctx.lineWidth = 5;
    ctx.beginPath();
    ctx.moveTo(toX(rs[0]), toY(rs[1]));
    ctx.lineTo(toX(re[0]), toY(re[1]));
    ctx.lineTo(toX(rw[0]), toY(rw[1]));
    ctx.stroke();

    // Head circle
    ctx.fillStyle = "#f8fafc";
    ctx.strokeStyle = "#334155";
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.arc(toX(head[0]), toY(head[1]), 26, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();

    // Hand indicator circles
    ctx.fillStyle = "#0284c7";
    ctx.beginPath();
    ctx.arc(toX(lw[0]), toY(lw[1]), 8, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = "#1d4ed8";
    ctx.beginPath();
    ctx.arc(toX(rw[0]), toY(rw[1]), 10, 0, Math.PI * 2);
    ctx.fill();

    // Update Player HUD cues
    if (playerSignCue) playerSignCue.textContent = `Sign: ${signName || "None"}`;
    if (playerHamnosysCue) playerHamnosysCue.textContent = `HamNoSys: ${hamnosys || "--"}`;
    if (playerTimelineCue) playerTimelineCue.textContent = `${currentSignIdx + 1} / ${playerQueue.length}`;

    // Frame Caption text on canvas
    if (frame.caption) {
      ctx.fillStyle = "#0f172a";
      ctx.font = "bold 13px sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(frame.caption, w / 2, h - 20);
    }
  }

  if (btnPlayerPlay) {
    btnPlayerPlay.addEventListener("click", () => {
      if (playerQueue.length > 0) {
        currentSignIdx = 0;
        currentFrameIdx = 0;
        startPlayerPlayback();
      }
    });
  }

  if (btnPlayerPause) {
    btnPlayerPause.addEventListener("click", () => {
      isPlaying = false;
      if (playerTimer) clearInterval(playerTimer);
      if (playerStatusBadge) playerStatusBadge.textContent = "Paused";
    });
  }

  if (btnPlayerStepPrev) {
    btnPlayerStepPrev.addEventListener("click", () => {
      if (playerQueue.length > 0) {
        currentSignIdx = Math.max(0, currentSignIdx - 1);
        currentFrameIdx = 0;
        const anim = playerQueue[currentSignIdx];
        if (anim && anim.frames) drawSignSkeleton(anim.frames[0], anim.sign, anim.hamnosys);
      }
    });
  }

  if (btnPlayerStepNext) {
    btnPlayerStepNext.addEventListener("click", () => {
      if (playerQueue.length > 0) {
        currentSignIdx = Math.min(playerQueue.length - 1, currentSignIdx + 1);
        currentFrameIdx = 0;
        const anim = playerQueue[currentSignIdx];
        if (anim && anim.frames) drawSignSkeleton(anim.frames[0], anim.sign, anim.hamnosys);
      }
    });
  }

  // 10. Session History Management & Transcript Exports
  const sessionIdLabel = document.getElementById("session-id-label");
  const sessionTurnCount = document.getElementById("session-turn-count");
  const btnSessionNew = document.getElementById("btn-session-new");
  const btnSessionClear = document.getElementById("btn-session-clear");
  const btnRefreshHistory = document.getElementById("btn-refresh-history");
  const sessionTurnsList = document.getElementById("session-turns-list");

  async function fetchSessionHistory() {
    try {
      const res = await fetch("/api/session/history");
      if (!res.ok) return;
      const data = await res.json();

      if (sessionIdLabel) sessionIdLabel.textContent = data.session_id || "--";
      if (sessionTurnCount) sessionTurnCount.textContent = data.total_turns || 0;

      if (!sessionTurnsList) return;
      sessionTurnsList.innerHTML = "";

      const turns = data.turns || [];
      if (turns.length === 0) {
        sessionTurnsList.innerHTML = '<p style="font-size: 0.75rem; color: var(--color-text-muted); text-align: center; padding: 12px;">No dialogue turns recorded in active session yet.</p>';
        return;
      }

      turns.slice().reverse().forEach((t) => {
        const isSigner = t.speaker && t.speaker.includes("Signer");
        const badgeColor = isSigner ? "var(--color-primary)" : "var(--color-success)";
        const card = document.createElement("div");
        card.style.cssText = "border: 1px solid var(--color-border-subtle); border-radius: var(--radius-sm); padding: 8px; background: var(--color-surface); font-size: 0.75rem;";
        card.innerHTML = `
          <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
            <span style="font-weight: 700; color: ${badgeColor};">${t.speaker} (Turn #${t.turn_id})</span>
            <span style="color: var(--color-text-muted); font-size: 0.6875rem;">${t.timestamp}</span>
          </div>
          <div>Input: <strong style="color: var(--color-text);">${t.input_content}</strong></div>
          <div style="margin-top: 2px;">Translation: <span style="color: var(--color-primary); font-weight: 600;">${t.translated_content}</span></div>
          <div style="margin-top: 4px; display: flex; gap: 8px; font-size: 0.6875rem; color: var(--color-text-muted);">
            <span>Modality: ${t.modality}</span>
            ${t.phonology_summary ? `<span>Phonology: ${t.phonology_summary}</span>` : ""}
          </div>
        `;
        sessionTurnsList.appendChild(card);
      });
    } catch (e) {
      console.warn("Failed to load session history:", e);
    }
  }

  if (btnSessionNew) {
    btnSessionNew.addEventListener("click", async () => {
      await fetch("/api/session/clear", { method: "POST" });
      fetchSessionHistory();
    });
  }

  if (btnSessionClear) {
    btnSessionClear.addEventListener("click", async () => {
      await fetch("/api/session/clear", { method: "POST" });
      fetchSessionHistory();
    });
  }

  if (btnRefreshHistory) {
    btnRefreshHistory.addEventListener("click", fetchSessionHistory);
  }

  // Initialization
  btnShowAlphabet.className = "btn btn-primary";
  loadLexicon();
  fetchEvaluationMetrics();
  fetchSessionHistory();
  setInterval(fetchLiveState, 200); // 5 Hz polling
});
