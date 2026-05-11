library(shiny)
library(shinyjs)
library(jsonlite)
library(shinyBS)

get_available_tasks <- function(python_bin = "python", script_dir = ".") {
  old_wd <- getwd()
  on.exit(setwd(old_wd), add = TRUE)

  resolved_dir <- tryCatch(
    normalizePath(script_dir, winslash = "/", mustWork = TRUE),
    error = function(e) NULL
  )
  if (is.null(resolved_dir)) return(NULL)

  setwd(resolved_dir)

  out <- tryCatch(
    system2(
      command = python_bin,
      args = c("list_tasks.py"),
      stdout = TRUE,
      stderr = TRUE
    ),
    error = function(e) NULL
  )

  if (is.null(out) || length(out) == 0) return(NULL)
  tryCatch(fromJSON(paste(out, collapse = "\n")), error = function(e) NULL)
}

get_available_horizons <- function(labels_dir) {
  manifest_path <- file.path(labels_dir, "manifest.json")
  if (!file.exists(manifest_path)) return(NULL)
  obj <- tryCatch(fromJSON(manifest_path), error = function(e) NULL)
  if (is.null(obj) || is.null(obj$horizons)) return(NULL)
  hs <- suppressWarnings(as.integer(obj$horizons))
  hs <- hs[!is.na(hs)]
  if (length(hs) == 0) return(NULL)
  sort(unique(hs))
}

ui <- fluidPage(
  useShinyjs(),

  titlePanel("Digital Twins (AMR)"),
  sidebarLayout(
    sidebarPanel(
      h4("Data & Task"),

      textInput(
        "script_dir",
        "Folder containing train_amr_dygformer.py and list_tasks.py",
        value = normalizePath(getwd(), winslash = "/", mustWork = FALSE)
      ),

      textInput(
        "data_folder",
        "Training data folder",
        value = "synthetic_amr_graphs_train"
      ),

      textInput(
        "test_folder",
        "Test data folder",
        value = "synthetic_amr_graphs_test"
      ),

      textInput(
        "out_dir",
        "Output folder",
        value = "training_outputs"
      ),

      hr(),
      checkboxInput(
        "use_task_hparams",
        "Let task override hparams (--use_task_hparams)",
        value = TRUE
      ),

      checkboxInput(
        "train_model",
        "Train model (if unchecked: use saved model only)",
        value = TRUE
      ),

      br(),
      actionButton("run_model", "Run model", class = "btn btn-primary"),

      br(),
      helpText("Training and test folders are now user-controlled and passed explicitly to the trainer."),

      br(),

      selectInput(
        "task",
        "Task",
        choices = c("Loading tasks…" = ""),
        selected = ""
      ),

      numericInput(
        "pred_horizon",
        "Prediction horizon H (days)",
        value = 7,
        min = 1,
        step = 1
      ),

      numericInput("T", "Window length T", value = 7, min = 1, step = 1),

      div(
        id = "hparam_panel",

        sliderInput(
          "sliding_step",
          "Sliding step",
          min = 1,
          max = 10,
          value = 1,
          step = 1
        ),

        hr(),
        h4("Model hyperparameters"),

        sliderInput(
          "hidden",
          "Hidden size",
          min = 16,
          max = 256,
          value = 64,
          step = 16
        ),

        numericInput("heads", "Transformer heads", value = 2, min = 1, step = 1),
        sliderInput("dropout", "Dropout", min = 0, max = 0.9, value = 0.2, step = 0.05),

        sliderInput(
          "transformer_layers",
          "Number of Transformer layers",
          min = 1,
          max = 10,
          value = 2,
          step = 1
        ),

        sliderInput(
          "sage_layers",
          "Number of GraphSAGE layers (depth)",
          min = 1,
          max = 10,
          value = 3,
          step = 1
        ),

        checkboxInput("use_cls", "Use [CLS] token", value = FALSE),

        hr(),
        h4("Training hyperparameters"),

        sliderInput(
          "batch_size",
          "Batch size",
          min = 8,
          max = 256,
          value = 16,
          step = 8
        ),

        numericInput("epochs", "Epochs", value = 20, min = 1, step = 1),
        numericInput("lr", "Learning rate", value = 1e-4, min = 1e-6, step = 1e-5),

        hr(),
        numericInput(
          "max_neighbors",
          "Legacy max_neighbors per node (edge-thinning; ignored if neighbor_sampling=TRUE)",
          value = 20,
          min = 0,
          step = 1
        )
      ),

      hr(),
      h4("Graph sampling (always user-controlled)"),

      checkboxInput(
        "neighbor_sampling",
        "Use true GraphSAGE neighbor sampling (--neighbor_sampling)",
        value = FALSE
      ),

      textInput(
        "num_neighbors",
        "num_neighbors per hop (comma-separated, e.g. 15,10)",
        value = "15,10"
      ),

      numericInput(
        "seed_count",
        "seed_count (number of seed nodes per graph)",
        value = 256,
        min = 0,
        step = 1
      ),

      selectInput(
        "seed_strategy",
        "seed_strategy",
        choices = c("random", "all"),
        selected = "random"
      ),

      numericInput(
        "seed_batch_size",
        "seed_batch_size (seeds per sampled subgraph batch)",
        value = 64,
        min = 1,
        step = 1
      ),

      numericInput(
        "max_sub_batches",
        "max_sub_batches (cap sampled subgraph batches per graph; 0 = no cap)",
        value = 4,
        min = 0,
        step = 1
      ),

      hr(),
      h4("Attention and translational exports"),

      sliderInput(
        "attn_top_k",
        "Top-K nodes to display",
        min = 10,
        max = 200,
        value = 10,
        step = 5
      ),
      selectInput(
        "attn_rank_by",
        "Rank nodes by",
        choices = c("abs_diff", "mean"),
        selected = "abs_diff"
      ),
      checkboxInput(
        "emit_translational_figures",
        "Export translational figures",
        value = TRUE
      ),
      checkboxInput(
        "fullgraph_attribution_pass",
        "Use full-graph attribution pass when neighbor sampling is enabled",
        value = TRUE
      ),
      numericInput(
        "translational_top_k",
        "Translational top-K",
        value = 20,
        min = 1,
        step = 1
      ),

      hr(),
      h4("Run guards and splitting"),

      numericInput(
        "split_seed",
        "Split seed",
        value = 0,
        min = 0,
        step = 1
      ),
      checkboxInput(
        "require_pt_metadata",
        "Require PT metadata (sim_id/day)",
        value = TRUE
      ),
      checkboxInput(
        "fail_on_noncontiguous",
        "Fail on non-contiguous temporal windows",
        value = TRUE
      ),

      hr(),
      h4("Training control"),

      checkboxInput(
        "early_stopping",
        "Use early stopping",
        value = FALSE
      ),
      numericInput("patience", "Early stopping patience", value = 7, min = 1, step = 1),
      numericInput("min_delta", "Minimum improvement delta", value = 1e-4, min = 0, step = 1e-5),
      checkboxInput(
        "save_best_only",
        "Save best checkpoint only",
        value = FALSE
      ),
      checkboxInput(
        "lr_scheduler_on_plateau",
        "Use ReduceLROnPlateau scheduler",
        value = FALSE
      ),
      numericInput("lr_scheduler_factor", "Scheduler factor", value = 0.5, min = 0.01, max = 0.99, step = 0.01),
      numericInput("lr_scheduler_patience", "Scheduler patience", value = 3, min = 1, step = 1),
      numericInput("lr_scheduler_min_lr", "Scheduler minimum LR", value = 1e-6, min = 0, step = 1e-6),

      hr(),
      textInput(
        "python_bin",
        "Python executable",
        value = "python"
      ),

      bsTooltip("script_dir", "Folder containing train_amr_dygformer.py and list_tasks.py. Commands are run from this folder.", placement = "right", trigger = "hover", options = list(container = "body")),
      bsTooltip("data_folder", "Folder containing the converted .pt training graphs.", placement = "right", trigger = "hover", options = list(container = "body")),
      bsTooltip("test_folder", "Explicit external test folder passed to --test_folder.", placement = "right", trigger = "hover", options = list(container = "body")),
      bsTooltip("out_dir", "Output directory passed to --out_dir for plots, summaries, and trained_model.pt.", placement = "right", trigger = "hover", options = list(container = "body")),
      bsTooltip("use_task_hparams", "If checked, the selected task overrides the model/training hyperparameters.", placement = "right", trigger = "hover", options = list(container = "body")),
      bsTooltip("train_model", "If unchecked, skips training and only runs evaluation using a saved model (if present).", placement = "right", trigger = "hover", options = list(container = "body")),
      bsTooltip("run_model", "Runs train_amr_dygformer.py with the arguments shown in Command preview.", placement = "right", trigger = "hover", options = list(container = "body")),
      bsTooltip("task", "Select the prediction task (classification or regression) as defined in tasks.py.", placement = "right", trigger = "hover", options = list(container = "body")),
      bsTooltip("pred_horizon", "Prediction horizon (days ahead). If the chosen task name ends with _h<digits>, the UI swaps it to _hH before calling Python. If the task has no _h suffix, this is ignored.", placement = "right", trigger = "hover", options = list(container = "body")),
      bsTooltip("T", "Temporal window length (number of days/time steps per input sequence).", placement = "right", trigger = "hover", options = list(container = "body")),

      bsTooltip("sliding_step", "Stride used to slide the T-length window through time (smaller = more overlap, more samples).", placement = "right", trigger = "hover", options = list(container = "body")),
      bsTooltip("hidden", "Transformer hidden dimension (embedding size). Larger can improve capacity but increases compute.", placement = "right", trigger = "hover", options = list(container = "body")),
      bsTooltip("heads", "Number of attention heads in the Transformer.", placement = "right", trigger = "hover", options = list(container = "body")),
      bsTooltip("dropout", "Dropout probability for regularisation.", placement = "right", trigger = "hover", options = list(container = "body")),
      bsTooltip("transformer_layers", "Number of stacked Transformer layers.", placement = "right", trigger = "hover", options = list(container = "body")),
      bsTooltip("sage_layers", "Number of GraphSAGE layers.", placement = "right", trigger = "hover", options = list(container = "body")),
      bsTooltip("use_cls", "If checked, uses a [CLS] token style pooling when supported.", placement = "right", trigger = "hover", options = list(container = "body")),

      bsTooltip("batch_size", "Training batch size.", placement = "right", trigger = "hover", options = list(container = "body")),
      bsTooltip("epochs", "Number of training epochs.", placement = "right", trigger = "hover", options = list(container = "body")),
      bsTooltip("lr", "Learning rate for the optimiser.", placement = "right", trigger = "hover", options = list(container = "body")),
      bsTooltip("max_neighbors", "Legacy edge-thinning cap per node. Disabled automatically when neighbor_sampling=TRUE.", placement = "right", trigger = "hover", options = list(container = "body")),

      bsTooltip("neighbor_sampling", "Enable true GraphSAGE neighbor sampling during training/eval.", placement = "right", trigger = "hover", options = list(container = "body")),
      bsTooltip("num_neighbors", "Comma-separated neighbor counts per hop (e.g., '15,10').", placement = "right", trigger = "hover", options = list(container = "body")),
      bsTooltip("seed_count", "Number of seed nodes sampled per graph when neighbor sampling is enabled.", placement = "right", trigger = "hover", options = list(container = "body")),
      bsTooltip("seed_strategy", "How to choose seed nodes: random subset or all nodes.", placement = "right", trigger = "hover", options = list(container = "body")),
      bsTooltip("seed_batch_size", "Seeds per sampled subgraph batch.", placement = "right", trigger = "hover", options = list(container = "body")),
      bsTooltip("max_sub_batches", "Maximum sampled subgraph batches per graph (0 means no cap).", placement = "right", trigger = "hover", options = list(container = "body")),

      bsTooltip("attn_top_k", "Number of top-ranked nodes to show in attention heatmap outputs.", placement = "right", trigger = "hover", options = list(container = "body")),
      bsTooltip("attn_rank_by", "Ranking statistic for attention nodes (abs_diff vs mean).", placement = "right", trigger = "hover", options = list(container = "body")),
      bsTooltip("emit_translational_figures", "Passes --emit_translational_figures to the trainer.", placement = "right", trigger = "hover", options = list(container = "body")),
      bsTooltip("fullgraph_attribution_pass", "Passes --fullgraph_attribution_pass so attribution exports can use full-graph inference even when training uses neighbor sampling.", placement = "right", trigger = "hover", options = list(container = "body")),
      bsTooltip("translational_top_k", "Top-K entities used in translational attribution exports.", placement = "right", trigger = "hover", options = list(container = "body")),

      bsTooltip("split_seed", "Seed controlling the trajectory-level train/validation split.", placement = "right", trigger = "hover", options = list(container = "body")),
      bsTooltip("require_pt_metadata", "Require sim_id/day metadata in .pt files.", placement = "right", trigger = "hover", options = list(container = "body")),
      bsTooltip("fail_on_noncontiguous", "Raise an error if non-contiguous temporal windows are detected.", placement = "right", trigger = "hover", options = list(container = "body")),

      bsTooltip("early_stopping", "Enable trainer-side early stopping.", placement = "right", trigger = "hover", options = list(container = "body")),
      bsTooltip("patience", "Number of epochs without improvement before early stopping.", placement = "right", trigger = "hover", options = list(container = "body")),
      bsTooltip("min_delta", "Minimum validation-loss improvement required to reset patience.", placement = "right", trigger = "hover", options = list(container = "body")),
      bsTooltip("save_best_only", "Save only the best checkpoint during training.", placement = "right", trigger = "hover", options = list(container = "body")),
      bsTooltip("lr_scheduler_on_plateau", "Enable ReduceLROnPlateau learning-rate scheduling.", placement = "right", trigger = "hover", options = list(container = "body")),
      bsTooltip("lr_scheduler_factor", "Factor by which LR is reduced when the scheduler triggers.", placement = "right", trigger = "hover", options = list(container = "body")),
      bsTooltip("lr_scheduler_patience", "Number of bad epochs before the LR scheduler triggers.", placement = "right", trigger = "hover", options = list(container = "body")),
      bsTooltip("lr_scheduler_min_lr", "Lower bound for the learning rate under scheduling.", placement = "right", trigger = "hover", options = list(container = "body")),
      bsTooltip("python_bin", "Python executable used to run list_tasks.py and train_amr_dygformer.py.", placement = "right", trigger = "hover", options = list(container = "body"))
    ),

    mainPanel(
      h4("Command preview"),
      verbatimTextOutput("cmd_preview"),

      h4("Run log"),
      verbatimTextOutput("run_log"),

      tabsetPanel(
        tabPanel(
          "Loss curves",
          br(),
          imageOutput("loss_curves_img", height = "100%")
        ),
        tabPanel(
          "Confusion matrix (val)",
          br(),
          imageOutput("cm_img", height = "100%")
        ),
        tabPanel(
          "ROC curve (val)",
          br(),
          imageOutput("roc_img", height = "100%")
        ),
        tabPanel(
          "Test confusion matrix",
          br(),
          imageOutput("cm_test_img", height = "100%")
        ),
        tabPanel(
          "Test ROC curve",
          br(),
          imageOutput("roc_test_img", height = "100%")
        ),
        tabPanel(
          "Attention heatmap (val)",
          br(),
          imageOutput("attn_heatmap_img", height = "100%")
        ),
        tabPanel(
          "Attention heatmap (test)",
          br(),
          imageOutput("attn_heatmap_test_img", height = "100%")
        )
      )
    )
  )
)

server <- function(input, output, session) {

  labels_dir_reactive <- reactive({
    script_dir <- tryCatch(
      normalizePath(input$script_dir, winslash = "/", mustWork = TRUE),
      error = function(e) NULL
    )
    if (is.null(script_dir)) return(NULL)
    file.path(script_dir, input$data_folder, "labels")
  })

  available_horizons <- reactive({
    labels_dir <- labels_dir_reactive()
    if (is.null(labels_dir)) return(NULL)
    get_available_horizons(labels_dir = labels_dir)
  })

  observe({
    hs <- available_horizons()
    if (!is.null(hs)) {
      Hmin <- min(hs)
      Hmax <- max(hs)
      Hcur <- as.integer(input$pred_horizon)
      if (is.na(Hcur)) Hcur <- Hmin
      if (Hcur < Hmin) Hcur <- Hmin
      if (Hcur > Hmax) Hcur <- Hmax
      updateNumericInput(session, "pred_horizon", min = Hmin, max = Hmax, value = Hcur)
    }
  })

  observe({
    if (isTRUE(input$use_task_hparams)) {
      shinyjs::disable("hparam_panel")
    } else {
      shinyjs::enable("hparam_panel")
    }
  })

  observe({
    if (isTRUE(input$neighbor_sampling)) {
      shinyjs::disable("max_neighbors")
    } else {
      shinyjs::enable("max_neighbors")
    }
  })

  observe({
    task_id <- as.character(input$task)
    has_h <- grepl("_h[0-9]+$", task_id)
    if (isTRUE(has_h)) {
      shinyjs::enable("pred_horizon")
    } else {
      shinyjs::disable("pred_horizon")
    }
  })

  log_text <- reactiveVal("")
  output$run_log <- renderText({ log_text() })

  run_dir <- reactiveVal("training_outputs")

  effective_task <- reactive({
    task_id <- as.character(input$task)
    if (is.na(task_id) || task_id == "") return("")
    if (grepl("_h[0-9]+$", task_id)) {
      H <- as.integer(input$pred_horizon)
      if (is.na(H) || H < 1) H <- 1
      return(sub("_h[0-9]+$", paste0("_h", H), task_id))
    }
    task_id
  })

  build_args <- reactive({
    args <- c(
      "train_amr_dygformer.py",
      "--data_folder", as.character(input$data_folder),
      "--test_folder", as.character(input$test_folder),
      "--task", effective_task(),
      "--T", as.character(input$T),
      "--out_dir", as.character(input$out_dir),
      "--emit_translational_figures", if (isTRUE(input$emit_translational_figures)) "true" else "false",
      "--fullgraph_attribution_pass", if (isTRUE(input$fullgraph_attribution_pass)) "true" else "false",
      "--translational_top_k", as.character(input$translational_top_k),
      "--split_seed", as.character(input$split_seed),
      "--require_pt_metadata", if (isTRUE(input$require_pt_metadata)) "true" else "false",
      "--fail_on_noncontiguous", if (isTRUE(input$fail_on_noncontiguous)) "true" else "false",
      "--early_stopping", if (isTRUE(input$early_stopping)) "true" else "false",
      "--patience", as.character(input$patience),
      "--min_delta", as.character(input$min_delta),
      "--save_best_only", if (isTRUE(input$save_best_only)) "true" else "false",
      "--lr_scheduler_on_plateau", if (isTRUE(input$lr_scheduler_on_plateau)) "true" else "false",
      "--lr_scheduler_factor", as.character(input$lr_scheduler_factor),
      "--lr_scheduler_patience", as.character(input$lr_scheduler_patience),
      "--lr_scheduler_min_lr", as.character(input$lr_scheduler_min_lr)
    )

    if (isTRUE(input$use_task_hparams)) {
      args <- c(args, "--use_task_hparams")
    } else {
      args <- c(
        args,
        "--sliding_step", as.character(input$sliding_step),
        "--hidden", as.character(input$hidden),
        "--heads", as.character(input$heads),
        "--dropout", as.character(input$dropout),
        "--transformer_layers", as.character(input$transformer_layers),
        "--sage_layers", as.character(input$sage_layers),
        "--batch_size", as.character(input$batch_size),
        "--epochs", as.character(input$epochs),
        "--lr", as.character(input$lr)
      )

      if (isTRUE(input$use_cls)) {
        args <- c(args, "--use_cls")
      }

      args <- c(args, "--max_neighbors", as.character(input$max_neighbors))
    }

    ns_flag <- if (isTRUE(input$neighbor_sampling)) "true" else "false"
    args <- c(args, "--neighbor_sampling", ns_flag)

    if (isTRUE(input$neighbor_sampling)) {
      args <- c(
        args,
        "--num_neighbors", as.character(input$num_neighbors),
        "--seed_count", as.character(input$seed_count),
        "--seed_strategy", as.character(input$seed_strategy),
        "--seed_batch_size", as.character(input$seed_batch_size),
        "--max_sub_batches", as.character(input$max_sub_batches),
        "--max_neighbors", "0"
      )
    }

    args <- c(
      args,
      "--attn_top_k", as.character(input$attn_top_k),
      "--attn_rank_by", as.character(input$attn_rank_by)
    )

    train_flag <- if (isTRUE(input$train_model)) "true" else "false"
    args <- c(args, "--train_model", train_flag)

    args
  })

  output$cmd_preview <- renderText({
    script_dir <- tryCatch(
      normalizePath(input$script_dir, winslash = "/", mustWork = TRUE),
      error = function(e) NULL
    )
    args <- build_args()
    quoted_args <- paste(vapply(args, shQuote, character(1)), collapse = " ")
    if (is.null(script_dir)) {
      return(paste(shQuote(input$python_bin), quoted_args))
    }
    paste("cd", shQuote(script_dir), "&&", shQuote(input$python_bin), quoted_args)
  })

  observeEvent(input$run_model, {
    task_id <- as.character(input$task)
    if (grepl("_h[0-9]+$", task_id)) {
      H <- as.integer(input$pred_horizon)
      hs <- available_horizons()
      if (!is.null(hs) && !(H %in% hs)) {
        log_text(paste0(
          log_text(),
          sprintf("ERROR: Requested horizon H=%d is not present in dataset horizons: %s\n", H, paste(hs, collapse = ","))
        ))
        return(NULL)
      }
      if (is.null(hs)) {
        log_text(paste0(log_text(), "WARNING: horizons manifest not found; training may fail if labels are missing.\n"))
      }
    }

    script_dir <- tryCatch(
      normalizePath(input$script_dir, winslash = "/", mustWork = TRUE),
      error = function(e) NULL
    )
    if (is.null(script_dir)) {
      log_text(paste0(log_text(), "ERROR: script_dir does not exist or is not accessible.\n"))
      return(NULL)
    }

    trainer_path <- file.path(script_dir, "train_amr_dygformer.py")
    tasks_path <- file.path(script_dir, "list_tasks.py")
    if (!file.exists(trainer_path)) {
      log_text(paste0(log_text(), "ERROR: train_amr_dygformer.py not found in script_dir.\n"))
      return(NULL)
    }
    if (!file.exists(tasks_path)) {
      log_text(paste0(log_text(), "WARNING: list_tasks.py not found in script_dir. Task refresh may fail.\n"))
    }

    quoted_args <- paste(vapply(build_args(), shQuote, character(1)), collapse = " ")
    full_cmd <- paste(shQuote(input$python_bin), quoted_args)

    log_text("")
    run_dir(as.character(input$out_dir))

    tryCatch({
      withProgress(message = "Running model...", value = 0, {
        epochs_total <- NA_real_
        test_batches_total <- NA_real_
        last_progress <- 0

        train_fraction <- 0.8
        test_fraction <- 0.2

        old_wd <- getwd()
        on.exit(setwd(old_wd), add = TRUE)
        setwd(script_dir)

        con <- pipe(full_cmd, open = "r")
        on.exit(close(con), add = TRUE)

        repeat {
          line <- readLines(con, n = 1, warn = FALSE)
          if (!length(line)) break

          isolate({
            current_log <- log_text()
            log_text(paste0(current_log, line, "\n"))
          })

          if (grepl("^DT_OUT_DIR\\s+", line)) {
            out_line <- sub("^DT_OUT_DIR\\s+", "", line)
            if (!is.na(out_line) && nzchar(out_line)) {
              isolate(run_dir(out_line))
            }
          } else if (grepl("^DT_PROGRESS_META", line)) {
            m <- regmatches(line, regexec("epochs=([0-9]+)", line))[[1]]
            if (length(m) >= 2) {
              epochs_total <- as.numeric(m[2])
            }
          } else if (grepl("^DT_PROGRESS_EPOCH", line)) {
            m <- regmatches(line, regexec("DT_PROGRESS_EPOCH\\s+([0-9]+)", line))[[1]]
            if (length(m) >= 2 && !is.na(epochs_total) && epochs_total > 0) {
              epoch_num <- as.numeric(m[2])
              new_progress <- min(train_fraction * epoch_num / epochs_total, train_fraction)
              incProgress(
                amount = new_progress - last_progress,
                detail = sprintf("Training epoch %d/%d", epoch_num, epochs_total)
              )
              last_progress <- new_progress
            }
          } else if (grepl("^DT_PROGRESS_TEST_META", line)) {
            m <- regmatches(line, regexec("batches=([0-9]+)", line))[[1]]
            if (length(m) >= 2) {
              test_batches_total <- as.numeric(m[2])
            }
          } else if (grepl("^DT_PROGRESS_TEST_BATCH", line)) {
            m <- regmatches(line, regexec("DT_PROGRESS_TEST_BATCH\\s+([0-9]+)", line))[[1]]
            if (length(m) >= 2 && !is.na(test_batches_total) && test_batches_total > 0) {
              batch_num <- as.numeric(m[2])
              new_progress <- train_fraction +
                min(test_fraction * batch_num / test_batches_total, test_fraction)
              incProgress(
                amount = new_progress - last_progress,
                detail = sprintf("Testing batch %d/%d", batch_num, test_batches_total)
              )
              last_progress <- new_progress
            }
          }
        }

        if (last_progress < 1) {
          incProgress(1 - last_progress, detail = "Done.")
        }
      })
    }, error = function(e) {
      isolate({
        current_log <- log_text()
        log_text(paste0(current_log, "Error calling Python: ", conditionMessage(e), "\n"))
      })
    })
  })

  output$loss_curves_img <- renderImage({
    file_path <- file.path(run_dir(), "loss_curves.png")
    if (!file.exists(file_path)) {
      return(list(src = "", filetype = "", alt = "Loss curves not found yet."))
    }
    list(src = file_path, contentType = "image/png", alt = "loss_curves.png")
  }, deleteFile = FALSE)

  output$cm_img <- renderImage({
    file_path <- file.path(run_dir(), "confusion_matrix.png")
    if (!file.exists(file_path)) {
      return(list(src = "", filetype = "", alt = "Confusion matrix not found yet."))
    }
    list(src = file_path, contentType = "image/png", alt = "confusion_matrix.png")
  }, deleteFile = FALSE)

  output$roc_img <- renderImage({
    file_path <- file.path(run_dir(), "roc_curve.png")
    if (!file.exists(file_path)) {
      return(list(src = "", filetype = "", alt = "ROC curve not found yet."))
    }
    list(src = file_path, contentType = "image/png", alt = "roc_curve.png")
  }, deleteFile = FALSE)

  output$cm_test_img <- renderImage({
    file_path <- file.path(run_dir(), "confusion_matrix_test.png")
    if (!file.exists(file_path)) {
      return(list(src = "", filetype = "", alt = "Test confusion matrix not found yet."))
    }
    list(src = file_path, contentType = "image/png", alt = "confusion_matrix_test.png")
  }, deleteFile = FALSE)

  output$roc_test_img <- renderImage({
    file_path <- file.path(run_dir(), "roc_curve_test.png")
    if (!file.exists(file_path)) {
      return(list(src = "", filetype = "", alt = "Test ROC curve not found yet."))
    }
    list(src = file_path, contentType = "image/png", alt = "roc_curve_test.png")
  }, deleteFile = FALSE)

  output$attn_heatmap_img <- renderImage({
    file_path <- file.path(run_dir(), "attention_heatmap.png")
    if (!file.exists(file_path)) {
      return(list(src = "", filetype = "", alt = "Attention heatmap (val) not found yet."))
    }
    list(
      src = file_path,
      contentType = "image/png",
      alt = "attention_heatmap.png",
      style = "max-width: 100%; height: auto;"
    )
  }, deleteFile = FALSE)

  output$attn_heatmap_test_img <- renderImage({
    file_path <- file.path(run_dir(), "attention_heatmap_test.png")
    if (!file.exists(file_path)) {
      return(list(src = "", filetype = "", alt = "Attention heatmap (test) not found yet."))
    }
    list(
      src = file_path,
      contentType = "image/png",
      alt = "attention_heatmap_test.png",
      style = "max-width: 100%; height: auto;"
    )
  }, deleteFile = FALSE)

  observe({
    tasks <- get_available_tasks(input$python_bin, input$script_dir)

    if (is.null(tasks) || nrow(tasks) == 0) {
      updateSelectInput(
        session, "task",
        choices = c("⚠️ No tasks found" = "")
      )
      return()
    }

    choices <- setNames(
      tasks$id,
      paste0(
        tasks$id,
        ifelse(tasks$is_classification, " [CLS]", " [REG]")
      )
    )

    updateSelectInput(
      session,
      "task",
      choices = choices,
      selected = tasks$id[1]
    )
  })
}

shinyApp(ui = ui, server = server)
