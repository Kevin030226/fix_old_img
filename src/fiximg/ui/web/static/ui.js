/* Native web UI for the image-restoration service.
 *
 * Served as a static asset from src/fiximg/ui/web/static/ui.js — never generated,
 * never interpolated. Everything the server knows arrives through the JSON block
 * with id "fiximg-bootstrap" and through the /ui/api/* routes, and every value is
 * written with textContent (or as an attribute this file sets) rather than by
 * building HTML from a string. That is the whole injection story for this page:
 * there is no innerHTML assignment anywhere below, so a stage name, a task id or
 * an error message containing markup renders as those characters.
 *
 * The page works without this file for reading (the shell and the sign-in form are
 * server-rendered); this adds upload, polling, the before/after slider and the
 * admin forms.
 */
(function () {
  "use strict";

  var bootstrapEl = document.getElementById("fiximg-bootstrap");
  var boot = {};
  try {
    boot = JSON.parse(bootstrapEl ? bootstrapEl.textContent : "{}") || {};
  } catch (err) {
    boot = {};
  }

  var CSRF = boot.csrf_token || "";
  var IS_ADMIN = !!boot.is_admin;
  var CAN_SUBMIT = boot.can_submit !== false;
  var MAX_UPLOAD_MB = Number(boot.max_upload_mb || 10);
  var POLL_MS = Number(boot.poll_ms || 1000);
  var POLL_TIMEOUT_MS = Number(boot.poll_timeout_ms || 30 * 60 * 1000);

  // ------------------------------------------------------------------ helpers
  function $(root, selector) {
    return root ? root.querySelector(selector) : null;
  }

  function $$(root, selector) {
    return root ? Array.prototype.slice.call(root.querySelectorAll(selector)) : [];
  }

  function el(tag, text, className) {
    var node = document.createElement(tag);
    if (text !== undefined && text !== null) {
      node.textContent = String(text);
    }
    if (className) {
      node.className = className;
    }
    return node;
  }

  function setText(node, text) {
    if (node) {
      node.textContent = text === undefined || text === null ? "" : String(text);
    }
  }

  function flash(message, kind) {
    var box = document.getElementById("flash");
    if (!box) {
      return;
    }
    box.textContent = "";
    if (!message) {
      return;
    }
    box.appendChild(el("div", message, kind === "ok" ? "ok" : "error"));
  }

  function show(node, visible) {
    if (!node) {
      return;
    }
    if (visible) {
      node.removeAttribute("hidden");
    } else {
      node.setAttribute("hidden", "hidden");
    }
  }

  function api(path, options) {
    options = options || {};
    var headers = options.headers || {};
    if (options.method && options.method !== "GET") {
      headers["X-CSRF-Token"] = CSRF;
    }
    if (options.json !== undefined) {
      headers["Content-Type"] = "application/json";
      options.body = JSON.stringify(options.json);
    }
    return fetch(path, {
      method: options.method || "GET",
      headers: headers,
      body: options.body,
      credentials: "same-origin",
    }).then(function (response) {
      var type = response.headers.get("content-type") || "";
      var parsed = type.indexOf("application/json") >= 0
        ? response.json().catch(function () { return {}; })
        : response.text().then(function (text) { return { message: text }; });
      return parsed.then(function (body) {
        if (!response.ok) {
          var detail = body && body.error ? body.error.message : (body && body.message);
          var failure = new Error(detail || ("Request failed (" + response.status + ")"));
          failure.status = response.status;
          failure.code = body && body.error ? body.error.code : undefined;
          failure.body = body;
          throw failure;
        }
        return body;
      });
    });
  }

  function announce(node, message, kind) {
    if (!node) {
      return;
    }
    node.textContent = message || "";
    node.className = kind === "ok" ? "ok" : (message ? "error" : "muted");
  }

  // -------------------------------------------------------------------- tabs
  function selectTab(name) {
    $$(document, "[data-tab]").forEach(function (button) {
      var selected = button.getAttribute("data-tab") === name;
      button.setAttribute("aria-selected", selected ? "true" : "false");
      var panel = document.getElementById("panel-" + button.getAttribute("data-tab"));
      show(panel, selected);
    });
    var target = document.getElementById("panel-" + name);
    if (target) {
      var focusable = target.querySelector("input, button, select, textarea, a");
      if (focusable && name !== "auto") {
        focusable.focus({ preventScroll: true });
      }
    }
    if (name === "history") {
      loadHistory(1);
    }
    if (name === "admin" && IS_ADMIN) {
      loadAdminSection("account");
    }
  }

  $$(document, "[data-tab]").forEach(function (button) {
    button.addEventListener("click", function () {
      selectTab(button.getAttribute("data-tab"));
    });
  });

  // Roving arrow-key navigation, so the tab strip is usable without a mouse.
  var tabStrip = document.querySelector('[role="tablist"][aria-label="Sections"]');
  if (tabStrip) {
    tabStrip.addEventListener("keydown", function (event) {
      var buttons = $$(tabStrip, "[data-tab]").filter(function (b) {
        return !b.hasAttribute("hidden");
      });
      var index = buttons.indexOf(document.activeElement);
      if (index < 0) {
        return;
      }
      var next = null;
      if (event.key === "ArrowRight") {
        next = buttons[(index + 1) % buttons.length];
      } else if (event.key === "ArrowLeft") {
        next = buttons[(index - 1 + buttons.length) % buttons.length];
      } else if (event.key === "Home") {
        next = buttons[0];
      } else if (event.key === "End") {
        next = buttons[buttons.length - 1];
      }
      if (next) {
        event.preventDefault();
        next.focus();
        selectTab(next.getAttribute("data-tab"));
      }
    });
  }

  function selectAdminSection(name) {
    $$(document, "[data-admin-tab]").forEach(function (button) {
      var selected = button.getAttribute("data-admin-tab") === name;
      button.setAttribute("aria-selected", selected ? "true" : "false");
      show(document.getElementById("admin-" + button.getAttribute("data-admin-tab")),
        selected);
    });
    loadAdminSection(name);
  }

  $$(document, "[data-admin-tab]").forEach(function (button) {
    button.addEventListener("click", function () {
      selectAdminSection(button.getAttribute("data-admin-tab"));
    });
  });

  // --------------------------------------------------------------- examples
  var EXAMPLES = {
    old: ["a", "b", "c", "d", "e", "f", "g", "h"].map(function (n) {
      return "examples/old/" + n + ".png";
    }),
    old_w_scratch: ["a", "b", "c", "d"].map(function (n) {
      return "examples/old_w_scratch/" + n + ".png";
    }),
    color: ["o1", "o2"].map(function (n) {
      return "examples/color/" + n + ".jpg";
    }),
  };

  function exampleList(name) {
    if (name === "all") {
      return EXAMPLES.old.concat(EXAMPLES.old_w_scratch, EXAMPLES.color);
    }
    return EXAMPLES[name] || [];
  }

  // Loading an example is a fetch of a same-origin static file, turned into a File
  // so the upload path is identical to a chosen file: one code path, one set of
  // validations. A user who clicks an example and a user who picks a file must not
  // reach the server differently.
  function loadExample(panel, path) {
    return fetch(path, { credentials: "same-origin" }).then(function (response) {
      if (!response.ok) {
        throw new Error("Could not load " + path);
      }
      return response.blob();
    }).then(function (blob) {
      var name = path.split("/").pop();
      var file = new File([blob], name, { type: blob.type || "image/png" });
      setInput(panel, file);
    }).catch(function (err) {
      announce($(panel, "[data-status]"), err.message);
    });
  }

  function renderExamples(panel) {
    var host = $(panel, "[data-examples]");
    if (!host) {
      return;
    }
    host.textContent = "";
    var kind = host.getAttribute("data-examples");
    exampleList(kind).forEach(function (path) {
      var button = el("button");
      button.type = "button";
      button.title = "Load " + path;
      var label = el("span", "Example " + path.split("/").pop(), "sr-only");
      button.appendChild(label);
      var img = el("img");
      img.src = path;
      img.alt = "";
      button.appendChild(img);
      button.addEventListener("click", function () {
        loadExample(panel, path);
      });
      host.appendChild(button);
    });
  }

  // ----------------------------------------------------------- upload state
  function setInput(panel, file) {
    var state = panelStates[panel.id] || (panelStates[panel.id] = {});
    state.file = file || null;
    setText($(panel, "[data-file-name]"), file ? file.name : "");
    var preview = $(panel, "[data-input-preview]");
    if (preview) {
      if (state.objectUrl) {
        URL.revokeObjectURL(state.objectUrl);
        state.objectUrl = null;
      }
      if (file) {
        state.objectUrl = URL.createObjectURL(file);
        preview.src = state.objectUrl;
        show(preview, true);
      } else {
        preview.removeAttribute("src");
        show(preview, false);
      }
    }
    if (panel.getAttribute("data-mode") === "auto") {
      refreshAnalysis(panel);
    }
  }

  var panelStates = {};

  function switchesOf(panel) {
    return $$(panel, "[data-switch]");
  }

  function switchValues(panel) {
    return switchesOf(panel).map(function (box) { return !!box.checked; });
  }

  function switchNames(panel) {
    return switchesOf(panel).map(function (box) {
      return box.getAttribute("data-switch");
    });
  }

  // The options payload mirrors the server's rule: a checked box on an
  // analysis-derived tab sends nothing (the analysis stays the authority), an
  // unchecked one sends false. The server re-derives the same mapping, so a client
  // that sent the wrong shape would still be corrected there rather than trusted.
  function optionsFor(panel) {
    var names = switchNames(panel);
    var values = switchValues(panel);
    if (!names.length) {
      return null;
    }
    var options = {};
    var analysisDecided = panel.getAttribute("data-mode") === "auto";
    names.forEach(function (name, index) {
      if (analysisDecided && (name === "face_enhance" || name === "auto_colorize")) {
        if (!values[index]) {
          options[name] = false;
        }
        return;
      }
      options[name] = values[index];
    });
    return options;
  }

  // ------------------------------------------------------------- analysis row
  function refreshAnalysis(panel) {
    var box = $(panel, "[data-analysis]");
    var state = panelStates[panel.id] || {};
    if (!box) {
      return;
    }
    if (!state.file) {
      setText(box, "");
      return;
    }
    var body = new FormData();
    body.append("mode", panel.getAttribute("data-mode"));
    body.append("image", state.file);
    body.append("options", JSON.stringify(optionsFor(panel) || {}));
    api("/ui/api/analysis", { method: "POST", body: body }).then(function (data) {
      setText(box, data.text || "");
    }).catch(function (err) {
      setText(box, "Analysis unavailable: " + err.message);
    });
  }

  // ------------------------------------------------------------ submit + poll
  function renderStages(panel, stages) {
    var list = $(panel, "[data-stages]");
    if (!list) {
      return;
    }
    list.textContent = "";
    (stages || []).forEach(function (stage) {
      var item = el("li");
      var text = stage.icon + " " + stage.name +
        (stage.duration_text ? " (" + stage.duration_text + ")" : "");
      item.appendChild(document.createTextNode(text));
      if (stage.reason) {
        item.appendChild(el("span", " \u2014 " + stage.reason, "why"));
      }
      list.appendChild(item);
    });
  }

  function renderProgress(panel, progress) {
    var fill = $(panel, "[data-bar]");
    if (fill) {
      fill.style.width = Math.max(0, Math.min(100, Number(progress) || 0)) + "%";
    }
  }

  function renderResult(panel, taskId) {
    if (!taskId) {
      return;
    }
    var state = panelStates[panel.id] || (panelStates[panel.id] = {});
    state.taskId = taskId;
    var url = "/ui/api/tasks/" + encodeURIComponent(taskId) + "/result";
    var image = $(panel, "[data-result-img]");
    if (image) {
      image.src = url;
      show(image, true);
    }
    var download = $(panel, "[data-download]");
    if (download) {
      download.href = url;
      download.setAttribute("download", taskId + ".png");
      show(download, true);
    }
    var compare = $(panel, "[data-compare]");
    if (compare && state.file && state.objectUrl) {
      $(compare, "[data-compare-before]").src = state.objectUrl;
      $(compare, "[data-compare-after]").src = url;
      show(compare, true);
      setCompare(compare, 50);
    }
  }

  function clearPanel(panel) {
    var state = panelStates[panel.id] || (panelStates[panel.id] = {});
    state.file = null;
    state.taskId = null;
    state.stopped = true;
    if (state.objectUrl) {
      URL.revokeObjectURL(state.objectUrl);
      state.objectUrl = null;
    }
    var fileInput = $(panel, "[data-upload]");
    if (fileInput) {
      fileInput.value = "";
    }
    setText($(panel, "[data-file-name]"), "");
    setText($(panel, "[data-status]"), "");
    setText($(panel, "[data-analysis]"), "");
    renderStages(panel, []);
    renderProgress(panel, 0);
    show($(panel, "[data-result-img]"), false);
    show($(panel, "[data-download]"), false);
    show($(panel, "[data-compare]"), false);
  }

  function pollTask(panel, taskId, startedAt) {
    var state = panelStates[panel.id] || (panelStates[panel.id] = {});
    if (state.stopped) {
      return Promise.resolve();
    }
    return api("/ui/api/tasks/" + encodeURIComponent(taskId)).then(function (frame) {
      if (state.stopped) {
        return;
      }
      setText($(panel, "[data-status]"), frame.text || "");
      renderProgress(panel, frame.progress);
      renderStages(panel, frame.stages);
      if (frame.completed) {
        if (frame.result_ready) {
          renderResult(panel, taskId);
        }
        return;
      }
      if (frame.failed || frame.cancelled) {
        return;
      }
      if (Date.now() - startedAt > POLL_TIMEOUT_MS) {
        setText($(panel, "[data-status]"),
          "Timed out waiting for the task to finish. It may still be queued; " +
          "check History for task " + taskId + ".");
        return;
      }
      return new Promise(function (resolve) {
        setTimeout(resolve, POLL_MS);
      }).then(function () {
        return pollTask(panel, taskId, startedAt);
      });
    }).catch(function (err) {
      setText($(panel, "[data-status]"), "Polling failed: " + err.message);
    });
  }

  function submit(panel) {
    var state = panelStates[panel.id] || (panelStates[panel.id] = {});
    var submitButton = $(panel, "[data-submit]");
    if (!CAN_SUBMIT) {
      flash("Change the initial password first (Admin \u2192 Account).");
      return;
    }
    if (!state.file) {
      setText($(panel, "[data-status]"), "Please choose an image first.");
      return;
    }
    if (state.file.size > MAX_UPLOAD_MB * 1024 * 1024) {
      setText($(panel, "[data-status]"),
        "That image is larger than the " + MAX_UPLOAD_MB + " MB upload limit.");
      return;
    }
    state.stopped = false;
    setText($(panel, "[data-status]"), "Submitting\u2026");
    renderStages(panel, []);
    renderProgress(panel, 0);
    show($(panel, "[data-result-img]"), false);
    show($(panel, "[data-download]"), false);
    if (submitButton) {
      submitButton.disabled = true;
    }
    var body = new FormData();
    body.append("mode", panel.getAttribute("data-mode"));
    body.append("image", state.file);
    var options = optionsFor(panel);
    if (options) {
      body.append("options", JSON.stringify(options));
    }
    api("/ui/api/tasks", { method: "POST", body: body }).then(function (data) {
      setText($(panel, "[data-status]"),
        "Queued as " + data.task_id + ". Waiting for the worker\u2026");
      return pollTask(panel, data.task_id, Date.now());
    }).catch(function (err) {
      setText($(panel, "[data-status]"),
        err.code === "RATE_LIMITED" ? err.message : ("\u26a0 " + err.message));
    }).then(function () {
      if (submitButton) {
        submitButton.disabled = false;
      }
    });
  }

  function wirePanel(panel) {
    renderExamples(panel);
    var fileInput = $(panel, "[data-upload]");
    if (fileInput) {
      fileInput.addEventListener("change", function () {
        setInput(panel, fileInput.files && fileInput.files[0]);
      });
    }
    switchesOf(panel).forEach(function (box) {
      box.addEventListener("change", function () {
        if (panel.getAttribute("data-mode") === "auto") {
          refreshAnalysis(panel);
        }
      });
    });
    var submitButton = $(panel, "[data-submit]");
    if (submitButton) {
      submitButton.addEventListener("click", function () { submit(panel); });
    }
    var clearButton = $(panel, "[data-clear]");
    if (clearButton) {
      clearButton.addEventListener("click", function () { clearPanel(panel); });
    }
    if (!CAN_SUBMIT && submitButton) {
      submitButton.disabled = true;
      submitButton.title = "Change the initial password first";
    }
  }

  $$(document, "[data-mode]").forEach(wirePanel);

  // ------------------------------------------------------- before/after slider
  function setCompare(compare, percent) {
    var clamped = Math.max(0, Math.min(100, percent));
    var after = $(compare, ".after");
    var handle = $(compare, ".handle");
    if (after) {
      after.style.width = clamped + "%";
    }
    if (handle) {
      handle.style.left = clamped + "%";
      handle.setAttribute("aria-valuenow", String(Math.round(clamped)));
    }
  }

  $$(document, ".compare").forEach(function (compare) {
    var handle = $(compare, ".handle");
    function fromEvent(event) {
      var box = compare.getBoundingClientRect();
      var x = (event.touches ? event.touches[0].clientX : event.clientX) - box.left;
      return box.width ? (x / box.width) * 100 : 50;
    }
    var dragging = false;
    compare.addEventListener("pointerdown", function (event) {
      dragging = true;
      setCompare(compare, fromEvent(event));
    });
    window.addEventListener("pointermove", function (event) {
      if (dragging) {
        setCompare(compare, fromEvent(event));
      }
    });
    window.addEventListener("pointerup", function () { dragging = false; });
    if (handle) {
      handle.addEventListener("keydown", function (event) {
        var current = Number(handle.getAttribute("aria-valuenow") || 50);
        if (event.key === "ArrowLeft") {
          setCompare(compare, current - 5);
          event.preventDefault();
        } else if (event.key === "ArrowRight") {
          setCompare(compare, current + 5);
          event.preventDefault();
        } else if (event.key === "Home") {
          setCompare(compare, 0);
          event.preventDefault();
        } else if (event.key === "End") {
          setCompare(compare, 100);
          event.preventDefault();
        }
      });
    }
  });

  // ------------------------------------------------------------------ history
  var historyState = { page: 1, pageSize: 20, selected: null, rows: [] };

  function renderHistoryTable(rows) {
    var body = document.getElementById("history-rows");
    if (!body) {
      return;
    }
    body.textContent = "";
    rows.forEach(function (row) {
      var tr = el("tr");
      tr.tabIndex = 0;
      tr.setAttribute("role", "button");
      tr.setAttribute("aria-selected", "false");
      tr.dataset.taskId = row.task_id;
      ["task_id", "task_type", "created", "duration", "status", "progress", "metrics"]
        .forEach(function (key, index) {
          var td = el("td", row[key], index === 0 ? "mono" : "");
          tr.appendChild(td);
        });
      function choose() {
        selectHistoryRow(row.task_id);
      }
      tr.addEventListener("click", choose);
      tr.addEventListener("keydown", function (event) {
        if (event.key === "Enter" || event.key === " ") {
          event.preventDefault();
          choose();
        }
      });
      body.appendChild(tr);
    });
  }

  function renderHistoryGallery(items) {
    var host = document.getElementById("history-gallery");
    if (!host) {
      return;
    }
    host.textContent = "";
    (items || []).forEach(function (item) {
      var button = el("button");
      button.type = "button";
      button.title = item.caption || item.task_id;
      button.setAttribute("aria-label", item.caption || item.task_id);
      var img = el("img");
      img.src = item.thumbnail_url;
      img.alt = item.caption || "";
      button.appendChild(img);
      button.addEventListener("click", function () {
        selectHistoryRow(item.task_id);
      });
      host.appendChild(button);
    });
  }

  function selectHistoryRow(taskId) {
    historyState.selected = taskId;
    $$(document, "#history-rows tr").forEach(function (tr) {
      tr.setAttribute("aria-selected", tr.dataset.taskId === taskId ? "true" : "false");
    });
    var rerun = document.getElementById("history-rerun");
    var retry = document.getElementById("history-retry");
    if (rerun) {
      rerun.disabled = !taskId;
    }
    if (retry) {
      retry.disabled = !taskId;
    }
    loadHistoryDetail(taskId);
  }

  function setImage(node, url) {
    if (!node) {
      return;
    }
    if (url) {
      node.src = url;
      show(node, true);
    } else {
      node.removeAttribute("src");
      show(node, false);
    }
  }

  function loadHistoryDetail(taskId) {
    if (!taskId) {
      return;
    }
    api("/ui/api/history/" + encodeURIComponent(taskId)).then(function (data) {
      setText(document.getElementById("history-detail"), data.detail || "");
      setImage(document.getElementById("history-input"), data.input_url);
      setImage(document.getElementById("history-result"), data.result_url);
      setImage(document.getElementById("history-mask"), data.mask_url);
      setImage(document.getElementById("history-face"), data.face_url);
      var compare = document.getElementById("history-compare");
      if (compare && data.input_url && data.result_url) {
        $(compare, "[data-compare-before]").src = data.input_url;
        $(compare, "[data-compare-after]").src = data.result_url;
        show(compare, true);
        setCompare(compare, 50);
      } else if (compare) {
        show(compare, false);
      }
      var download = document.getElementById("history-download");
      if (download) {
        if (data.download_url) {
          download.href = data.download_url;
          download.setAttribute("download", taskId + ".png");
          show(download, true);
        } else {
          download.removeAttribute("href");
          show(download, false);
        }
      }
    }).catch(function (err) {
      setText(document.getElementById("history-detail"), "Could not load: " + err.message);
    });
  }

  function loadHistory(page) {
    var sizeSelect = document.getElementById("history-page-size");
    var pageSize = Number(sizeSelect ? sizeSelect.value : 20) || 20;
    var wanted = Math.max(1, Number(page) || 1);
    return api("/ui/api/history?page=" + wanted + "&page_size=" + pageSize)
      .then(function (data) {
        historyState.page = data.page;
        historyState.pageSize = data.page_size;
        historyState.rows = data.rows;
        setText(document.getElementById("history-page-label"), data.page_label);
        setText(document.getElementById("history-status"), data.status);
        renderHistoryTable(data.rows);
        renderHistoryGallery(data.thumbnails);
        if (historyState.selected) {
          selectHistoryRow(historyState.selected);
        }
      }).catch(function (err) {
        setText(document.getElementById("history-status"), "Could not load history: " + err.message);
      });
  }

  var historyRefresh = document.getElementById("history-refresh");
  if (historyRefresh) {
    historyRefresh.addEventListener("click", function () { loadHistory(historyState.page); });
  }
  var historyPrev = document.getElementById("history-prev");
  if (historyPrev) {
    historyPrev.addEventListener("click", function () {
      loadHistory(Math.max(1, historyState.page - 1));
    });
  }
  var historyNext = document.getElementById("history-next");
  if (historyNext) {
    historyNext.addEventListener("click", function () {
      loadHistory(historyState.page + 1);
    });
  }
  var historySize = document.getElementById("history-page-size");
  if (historySize) {
    historySize.addEventListener("change", function () { loadHistory(1); });
  }
  var historyRerun = document.getElementById("history-rerun");
  if (historyRerun) {
    historyRerun.addEventListener("click", function () {
      if (!historyState.selected) {
        return;
      }
      api("/ui/api/history/" + encodeURIComponent(historyState.selected) + "/rerun",
        { method: "POST", json: {} })
        .then(function (data) {
          setText(document.getElementById("history-status"), data.message);
          loadHistory(historyState.page);
        }).catch(function (err) {
          setText(document.getElementById("history-status"), err.message);
        });
    });
  }
  var historyRetry = document.getElementById("history-retry");
  if (historyRetry) {
    historyRetry.addEventListener("click", function () {
      if (!historyState.selected) {
        return;
      }
      api("/ui/api/history/" + encodeURIComponent(historyState.selected) + "/retry",
        { method: "POST", json: {} })
        .then(function (data) {
          setText(document.getElementById("history-status"), data.message);
          loadHistory(historyState.page);
        }).catch(function (err) {
          setText(document.getElementById("history-status"), err.message);
        });
    });
  }

  // -------------------------------------------------------------------- admin
  function loadAdminSection(name) {
    if (!IS_ADMIN) {
      return;
    }
    if (name === "tasks") {
      loadAdminHistory();
    } else if (name === "archive") {
      loadArchiveList();
    } else if (name === "users") {
      loadUsers();
    } else if (name === "models") {
      loadModels();
    }
  }

  function fillTable(body, rows, cellFor) {
    if (!body) {
      return;
    }
    body.textContent = "";
    rows.forEach(function (row) {
      var tr = el("tr");
      cellFor(row).forEach(function (value) {
        tr.appendChild(el("td", value));
      });
      body.appendChild(tr);
    });
  }

  function loadAdminHistory() {
    return api("/ui/api/admin/history").then(function (data) {
      fillTable(document.getElementById("admin-history-rows"), data.rows, function (row) {
        return [row.timestamp, row.user, row.type, row.status, row.psnr,
          row.ssim, row.mae, row.input_name];
      });
      setText(document.getElementById("admin-stats"), data.stats);
    }).catch(function (err) {
      flash("Could not load the task history: " + err.message);
    });
  }

  var adminHistoryRefresh = document.getElementById("admin-history-refresh");
  if (adminHistoryRefresh) {
    adminHistoryRefresh.addEventListener("click", loadAdminHistory);
  }
  var adminClear = document.getElementById("admin-clear-history");
  if (adminClear) {
    adminClear.addEventListener("click", function () {
      api("/ui/api/admin/history", { method: "DELETE" }).then(function (data) {
        announce(document.getElementById("admin-clear-status"), data.message, "ok");
        loadAdminHistory();
      }).catch(function (err) {
        announce(document.getElementById("admin-clear-status"), err.message);
      });
    });
  }

  function loadArchiveList() {
    return api("/ui/api/admin/archive").then(function (data) {
      var select = document.getElementById("archive-select");
      if (!select) {
        return;
      }
      select.textContent = "";
      (data.records || []).forEach(function (record) {
        var option = el("option", record.label);
        option.value = record.task_id;
        select.appendChild(option);
      });
      if (!data.records || !data.records.length) {
        var option = el("option", "No records");
        option.value = "";
        select.appendChild(option);
      }
      loadArchiveEntry(select.value);
    }).catch(function (err) {
      flash("Could not load the archive: " + err.message);
    });
  }

  function loadArchiveEntry(taskId) {
    if (!taskId) {
      setText(document.getElementById("archive-detail"), "");
      setImage(document.getElementById("archive-input"), null);
      setImage(document.getElementById("archive-output"), null);
      return;
    }
    api("/ui/api/admin/archive/" + encodeURIComponent(taskId)).then(function (data) {
      setText(document.getElementById("archive-detail"), data.detail);
      setImage(document.getElementById("archive-input"), data.input_url);
      setImage(document.getElementById("archive-output"), data.output_url);
    }).catch(function (err) {
      setText(document.getElementById("archive-detail"), "Could not load: " + err.message);
    });
  }

  var archiveRefresh = document.getElementById("archive-refresh");
  if (archiveRefresh) {
    archiveRefresh.addEventListener("click", loadArchiveList);
  }
  var archiveSelect = document.getElementById("archive-select");
  if (archiveSelect) {
    archiveSelect.addEventListener("change", function () {
      loadArchiveEntry(archiveSelect.value);
    });
  }

  function loadUsers() {
    return api("/ui/api/admin/users").then(function (data) {
      fillTable(document.getElementById("users-rows"), data.users, function (user) {
        return [user.username, user.role];
      });
      [document.getElementById("user-edit-select"),
        document.getElementById("user-delete-select")].forEach(function (select) {
        if (!select) {
          return;
        }
        var previous = select.value;
        select.textContent = "";
        data.users.forEach(function (user) {
          var option = el("option", user.username);
          option.value = user.username;
          select.appendChild(option);
        });
        if (previous) {
          select.value = previous;
        }
      });
    }).catch(function (err) {
      flash("Could not load the users: " + err.message);
    });
  }

  var usersRefresh = document.getElementById("users-refresh");
  if (usersRefresh) {
    usersRefresh.addEventListener("click", loadUsers);
  }

  function formJson(form) {
    var data = {};
    Array.prototype.forEach.call(form.elements, function (field) {
      if (field.name) {
        data[field.name] = field.value;
      }
    });
    return data;
  }

  var userAdd = document.getElementById("user-add-form");
  if (userAdd) {
    userAdd.addEventListener("submit", function (event) {
      event.preventDefault();
      api("/ui/api/admin/users", { method: "POST", json: formJson(userAdd) })
        .then(function (data) {
          announce(document.getElementById("user-add-status"), data.message, "ok");
          userAdd.reset();
          loadUsers();
        }).catch(function (err) {
          announce(document.getElementById("user-add-status"), err.message);
        });
    });
  }

  var userEdit = document.getElementById("user-edit-form");
  if (userEdit) {
    userEdit.addEventListener("submit", function (event) {
      event.preventDefault();
      var payload = formJson(userEdit);
      var username = payload.username;
      var body = {};
      if (payload.password) {
        body.password = payload.password;
      }
      if (payload.role) {
        body.role = payload.role;
      }
      if (!username) {
        announce(document.getElementById("user-edit-status"), "Please select a user");
        return;
      }
      if (!body.password && !body.role) {
        announce(document.getElementById("user-edit-status"),
          "Provide a new password and/or a new role");
        return;
      }
      api("/ui/api/admin/users/" + encodeURIComponent(username),
        { method: "PATCH", json: body })
        .then(function (data) {
          announce(document.getElementById("user-edit-status"), data.message, "ok");
          userEdit.reset();
          loadUsers();
        }).catch(function (err) {
          announce(document.getElementById("user-edit-status"), err.message);
        });
    });
  }

  var userDelete = document.getElementById("user-delete-form");
  if (userDelete) {
    userDelete.addEventListener("submit", function (event) {
      event.preventDefault();
      var username = formJson(userDelete).username;
      if (!username) {
        announce(document.getElementById("user-delete-status"), "Please select a user");
        return;
      }
      api("/ui/api/admin/users/" + encodeURIComponent(username), { method: "DELETE" })
        .then(function (data) {
          announce(document.getElementById("user-delete-status"), data.message, "ok");
          userDelete.reset();
          loadUsers();
        }).catch(function (err) {
          announce(document.getElementById("user-delete-status"), err.message);
        });
    });
  }

  function loadModels() {
    return api("/ui/api/admin/models").then(function (data) {
      var body = document.getElementById("models-rows");
      if (!body) {
        return;
      }
      body.textContent = "";
      (data.models || []).forEach(function (model) {
        var tr = el("tr");
        [model.name, model.version, model.status, model.device,
          model.weight_present ? "present" : "missing"].forEach(function (value) {
          tr.appendChild(el("td", value));
        });
        var actions = el("td");
        ["warmup", "unload"].forEach(function (action) {
          var button = el("button", action, "btn secondary");
          button.type = "button";
          button.addEventListener("click", function () {
            api("/ui/api/admin/models/" + encodeURIComponent(model.name) + "/" + action,
              { method: "POST", json: {} })
              .then(function () {
                announce(document.getElementById("models-status"),
                  model.name + ": " + action + " requested", "ok");
                loadModels();
              }).catch(function (err) {
                announce(document.getElementById("models-status"), err.message);
              });
          });
          actions.appendChild(button);
        });
        tr.appendChild(actions);
        body.appendChild(tr);
      });
    }).catch(function (err) {
      announce(document.getElementById("models-status"),
        "Could not load the models: " + err.message);
    });
  }

  var modelsRefresh = document.getElementById("models-refresh");
  if (modelsRefresh) {
    modelsRefresh.addEventListener("click", loadModels);
  }

  var passwordForm = document.getElementById("password-form");
  if (passwordForm) {
    passwordForm.addEventListener("submit", function (event) {
      event.preventDefault();
      var payload = formJson(passwordForm);
      api("/ui/api/session/password", { method: "POST", json: payload })
        .then(function (data) {
          announce(document.getElementById("password-status"), data.message, "ok");
          passwordForm.reset();
          CAN_SUBMIT = true;
          show(document.getElementById("must-change"), false);
          $$(document, "[data-submit]").forEach(function (button) {
            button.disabled = false;
            button.removeAttribute("title");
          });
        }).catch(function (err) {
          announce(document.getElementById("password-status"), err.message);
        });
    });
  }

  // ----------------------------------------------------------------- sign out
  var signOut = document.getElementById("sign-out");
  if (signOut) {
    signOut.addEventListener("click", function () {
      api("/ui/api/session/logout", { method: "POST", json: {} }).then(function () {
        window.location.assign("/");
      }).catch(function () {
        window.location.assign("/");
      });
    });
  }

  // -------------------------------------------------------------------- start
  if (boot.must_change_password) {
    show(document.getElementById("must-change"), true);
  }
  if (IS_ADMIN) {
    show(document.getElementById("tab-admin"), true);
    selectTab("admin");
  }
})();
