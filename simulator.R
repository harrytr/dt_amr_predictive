library(shiny)
library(shinydashboard)
library(shinyjs)
library(bslib)
library(shinyBS)

ui <- dashboardPage(
  skin = "blue",
  dashboardHeader(title = "AMR Hospital Simulator"),
  dashboardSidebar(
    sidebarMenu(
      menuItem("Run", tabName = "run", icon = icon("play")),
      menuItem("Logs", tabName = "logs", icon = icon("terminal"))
    )
  ),
  dashboardBody(
    useShinyjs(),
    tabItems(
      tabItem(
        tabName = "run",
        fluidRow(
          box(
            title = "Simulation settings",
            width = 12,
            status = "primary",
            solidHeader = TRUE,
            
            fluidRow(
              column(4, numericInput("seed", "Seed", 44, min = 1)),
              column(4, numericInput("num_days", "Days", 90, min = 1)),
              column(4, numericInput("num_regions", "Regions", 1, min = 1))
            ),
            
            fluidRow(
              column(4, numericInput("num_wards", "Wards", 30, min = 1)),
              column(4, numericInput("num_patients", "Patients", 200, min = 1)),
              column(4, numericInput("num_staff", "Staff", 500, min = 1))
            ),
            
            hr(),
            h4("Superspreader injection"),
            
            fluidRow(
              column(4, checkboxInput("enable_superspreader", "Enable", value = FALSE)),
              column(
                4,
                selectInput(
                  "superspreader_mode",
                  "How to choose staff",
                  choices = c("Manual" = "manual", "Dropdown" = "dropdown", "Random" = "random"),
                  selected = "manual"
                )
              ),
              column(4, textInput("superspreader_staff_manual", "Manual Staff ID (e.g., s70)", value = ""))
            ),
            
            fluidRow(
              column(
                4,
                selectInput(
                  "superspreader_state",
                  "Superspreader initial state (day 0)",
                  choices = c("U", "CS", "CR", "IS", "IR"),
                  selected = "IR"
                )
              ),
              column(8, tags$small("Tip: choose IR to guarantee an infectious, resistant index case."))
            ),
            
            fluidRow(
              column(
                6,
                selectInput(
                  "superspreader_staff_dropdown",
                  "Dropdown Staff ID",
                  choices = c("s0"),
                  selected = "s0"
                )
              ),
              column(6, verbatimTextOutput("superspreader_resolved"))
            ),
            
            fluidRow(
              column(4, numericInput("superspreader_staff_contacts", "Extra staff contacts/day", value = 50, min = 0)),
              column(4, numericInput("superspreader_patient_frac_mult", "Patient contact multiplier", value = 3.0, min = 0.0, step = 0.5)),
              column(4, numericInput("superspreader_patient_min_add", "Add patient contacts/ward", value = 10, min = 0))
            ),
            
            fluidRow(
              column(4, numericInput("superspreader_edge_weight_mult", "Edge weight multiplier", value = 1.5, min = 0.1, step = 0.1)),
              column(4, numericInput("superspreader_start_day", "Start day", value = 1, min = 1)),
              column(4, numericInput("superspreader_end_day", "End day", value = 9999, min = 1))
            ),
            
            hr(),
            h4("Importation, turnover, and screening (new)"),
            
            # ------------------------------------------------------------
            # Dynamic census / turnover (optional)
            # ------------------------------------------------------------
            fluidRow(
              column(4, checkboxInput("enable_turnover", "Enable admissions/discharges", value = TRUE)),
              column(4, numericInput("daily_discharge_frac", "Daily discharge fraction", value = 0.2, min = 0.0, max = 1.0, step = 0.01)),
              column(4, numericInput("daily_discharge_min_per_ward", "Min discharges per ward/day", value = 0, min = 0, step = 1))
            ),
            fluidRow(
              column(4, numericInput("p_admit_import_cs", "P(admit import CS)", value = 0.15, min = 0.0, max = 1.0, step = 0.01)),
              column(4, numericInput("p_admit_import_cr", "P(admit import CR)", value = 0.1, min = 0.0, max = 1.0, step = 0.01)),
              column(4, tags$small("Only applied if admissions/discharges enabled in the simulator."))
            ),
            
            # ------------------------------------------------------------
            # Seasonal admission importation (optional; only relevant if turnover enabled)
            # ------------------------------------------------------------
            hr(),
            h4("Seasonal admission importation"),
            
            fluidRow(
              column(4, checkboxInput("enable_admit_import_seasonality", "Enable seasonal admission importation", value = TRUE)),
              column(
                4,
                selectInput(
                  "admit_import_seasonality",
                  "Seasonality mode",
                  choices = c("None" = "none", "Sinusoid" = "sinusoid", "Piecewise" = "piecewise", "Shock" = "shock"),
                  selected = "none"
                )
              ),
              column(4, checkboxInput("admit_import_show_advanced", "Show advanced controls", value = FALSE))
            ),
            
            # Period is meaningful for sinusoid/piecewise, not shock
            fluidRow(
              column(4, numericInput("admit_import_period_days", "Period (days)", value = 7, min = 1, step = 1)),
              column(8, tags$small("Sinusoid/piecewise use Period. Shock ignores Period."))
            ),
            
            # --- SINUSOID CONTROLS (enabled only in sinusoid) ---
            conditionalPanel(
              condition = "input.enable_admit_import_seasonality && input.admit_import_seasonality == 'sinusoid'",
              fluidRow(
                column(4, numericInput("admit_import_amp", "Amplitude (0..0.99)", value = 0.5, min = 0.0, max = 0.99, step = 0.05)),
                column(4, numericInput("admit_import_phase_day", "Phase day (peak shift)", value = 0, min = 0, step = 1)),
                column(4, tags$small("m(d)=1 + amp·sin(2π(t-ϕ)/P)."))
              )
            ),
            
            # --- PIECEWISE CONTROLS (enabled only in piecewise) ---
            conditionalPanel(
              condition = "input.enable_admit_import_seasonality && input.admit_import_seasonality == 'piecewise'",
              tagList(
                fluidRow(
                  column(4, numericInput("admit_import_high_start_day", "High season start (1..period)", value = 1, min = 1, step = 1)),
                  column(4, numericInput("admit_import_high_end_day", "High season end (1..period)", value = 90, min = 1, step = 1)),
                  column(4, tags$small("Wrap-around if start > end."))
                ),
                fluidRow(
                  column(4, numericInput("admit_import_high_mult", "High multiplier", value = 1.5, min = 0.0, step = 0.1)),
                  column(4, numericInput("admit_import_low_mult", "Low multiplier", value = 1.0, min = 0.0, step = 0.1)),
                  column(4, tags$small("m(d)=high_mult in-season else low_mult."))
                )
              )
            ),
            
            # --- SHOCK CONTROLS (enabled only in shock) ---
            conditionalPanel(
              condition = "input.enable_admit_import_seasonality && input.admit_import_seasonality == 'shock'",
              tagList(
                fluidRow(
                  column(4, numericInput("admit_import_shock_min_days", "Shock min duration (days)", value = 7, min = 1, step = 1)),
                  column(4, numericInput("admit_import_shock_max_days", "Shock max duration (days)", value = 30, min = 1, step = 1)),
                  column(4, tags$small("Single random shock window per region/trajectory."))
                ),
                fluidRow(
                  column(4, numericInput("admit_import_shock_mult_min", "Shock multiplier min", value = 1.5, min = 0.0, step = 0.1)),
                  column(4, numericInput("admit_import_shock_mult_max", "Shock multiplier max", value = 5.0, min = 0.0, step = 0.1)),
                  column(4, tags$small("Multiplier drawn uniformly in [min,max]."))
                )
              )
            ),
            
            # --- CAPS: make them always available but greyed out unless advanced OR seasonality enabled ---
            conditionalPanel(
              condition = "input.enable_admit_import_seasonality && input.admit_import_seasonality != 'none'",
              fluidRow(
                column(4, numericInput("admit_import_pmax_cs", "Cap p(CS) after scaling", value = 1.0, min = 0.0, max = 1.0, step = 0.05)),
                column(4, numericInput("admit_import_pmax_cr", "Cap p(CR) after scaling", value = 1.0, min = 0.0, max = 1.0, step = 0.05)),
                column(4, tags$small("Safety caps; keep at 1.0 unless you have a reason."))
              )
            ),
            
            # Advanced diagnostics/help text (optional)
            conditionalPanel(
              condition = "input.enable_admit_import_seasonality && input.admit_import_show_advanced",
              tags$div(
                style = "padding:8px; border:1px solid #ddd; border-radius:6px; background:#fafafa;",
                tags$strong("Advanced note: "),
                "Period/phase and caps change only admission importation scaling. Base admission probabilities remain set by P(admit import CS/CR)."
              )
            ),
            
            # ------------------------------------------------------------
            # Screening/observability (optional)
            # ------------------------------------------------------------
            fluidRow(
              column(4, checkboxInput("enable_screening_controls", "Enable screening controls", value = TRUE)),
              column(4, numericInput("screen_every_k_days", "Screen every k days (k>=1)", value = 7, min = 1, step = 1)),
              column(4, numericInput("screen_result_delay_days", "Result delay (days)", value = 0, min = 0, step = 1))
            ),
            fluidRow(
              column(4, checkboxInput("screen_on_admission", "Screen on admission", value = FALSE)),
              column(4, checkboxInput("persist_observations", "Persist observed status across days", value = TRUE)),
              column(4, tags$small("If disabled, observed status may reset daily (legacy behavior)."))
            ),
            
            # ------------------------------------------------------------
            # Cross-ward mixing (optional)
            # ------------------------------------------------------------
            fluidRow(
              column(4, checkboxInput("override_staff_wards_per_staff", "Override staff multi-ward assignment", value = FALSE)),
              column(4, numericInput("staff_wards_per_staff", "Wards per staff", value = 2, min = 1, step = 1)),
              column(4, tags$small("Higher values increase cross-ward coupling via staff."))
            ),
            
            hr(),
            selectInput(
              "dataset",
              "Dataset to generate",
              choices = c("learn", "test"),
              selected = "learn"
            ),
            
            textInput(
              "label_horizons",
              "Label horizons (days, comma-separated)",
              value = "3,7,14"
            ),
            
            hr(),
            textInput(
              "python_bin",
              "Python executable",
              value = "python"
            ),
            checkboxInput(
              "export_gif",
              "Export GIF animation",
              value = TRUE
            ),
            
            textInput(
              "script_dir",
              "Folder containing generate_amr_data.py and convert_to_pt.py",
              value = normalizePath(getwd(), winslash = "/", mustWork = FALSE)
            ),

            hr(),
            h4("PT conversion settings"),

            fluidRow(
              column(
                3,
                selectInput(
                  "state_mode",
                  "State mode",
                  choices = c(
                    "ground_truth" = "ground_truth",
                    "partial_observation" = "partial_observation"
                  ),
                  selected = "ground_truth"
                )
              ),
              column(3, numericInput("conv_workers", "Conversion workers (0 = serial)", value = 0, min = 0, step = 1)),
              column(3, checkboxInput("keep_graphml", "Keep GraphML after conversion", value = TRUE)),
              column(3, checkboxInput("use_pt_out_dir", "Also archive PT files elsewhere", value = FALSE))
            ),

            conditionalPanel(
              condition = "input.use_pt_out_dir",
              textInput(
                "pt_out_dir",
                "PT archive folder",
                value = ""
              )
            ),

            br(),
            actionButton(
              "run_btn",
              "▶ Run simulation",
              icon = icon("play"),
              class = "btn-success btn-lg"
            ),
            
            # ------------------------------------------------------------
            # Tooltips (Option A: shinyBS)
            # ------------------------------------------------------------
            bsTooltip("seed", "Random seed for reproducibility.", placement = "right", trigger = "hover", options = list(container = "body")),
            bsTooltip("num_days", "Number of simulated days (one GraphML snapshot per day).", placement = "right", trigger = "hover", options = list(container = "body")),
            bsTooltip("num_regions", "Independent regions/hospital systems simulated.", placement = "right", trigger = "hover", options = list(container = "body")),
            bsTooltip("num_wards", "Number of wards in the hospital.", placement = "right", trigger = "hover", options = list(container = "body")),
            bsTooltip("num_patients", "Total number of patient nodes.", placement = "right", trigger = "hover", options = list(container = "body")),
            bsTooltip("num_staff", "Total number of staff nodes.", placement = "right", trigger = "hover", options = list(container = "body")),
            
            bsTooltip("enable_superspreader", "Enable a designated staff member with inflated contact rate.", placement = "right", trigger = "hover", options = list(container = "body")),
            bsTooltip("superspreader_mode", "Manual / Dropdown / Random (random uses Seed).", placement = "right", trigger = "hover", options = list(container = "body")),
            bsTooltip("superspreader_staff_manual", "Manual staff ID (e.g., s70).", placement = "right", trigger = "hover", options = list(container = "body")),
            bsTooltip("superspreader_state", "Initial AMR state on day 0 for the superspreader.", placement = "right", trigger = "hover", options = list(container = "body")),
            bsTooltip("superspreader_staff_dropdown", "Choose superspreader staff ID from valid staff IDs.", placement = "right", trigger = "hover", options = list(container = "body")),
            bsTooltip("superspreader_staff_contacts", "Extra staff-to-staff contacts/day for superspreader.", placement = "right", trigger = "hover", options = list(container = "body")),
            bsTooltip("superspreader_patient_frac_mult", "Multiplier for superspreader patient contacts vs baseline.", placement = "right", trigger = "hover", options = list(container = "body")),
            bsTooltip("superspreader_patient_min_add", "Minimum additional patient contacts/ward from superspreader.", placement = "right", trigger = "hover", options = list(container = "body")),
            bsTooltip("superspreader_edge_weight_mult", "Multiplier applied to transmission weights on superspreader edges.", placement = "right", trigger = "hover", options = list(container = "body")),
            bsTooltip("superspreader_start_day", "Day (1-based) when superspreader effect begins.", placement = "right", trigger = "hover", options = list(container = "body")),
            bsTooltip("superspreader_end_day", "Day (1-based) when superspreader effect ends.", placement = "right", trigger = "hover", options = list(container = "body")),
            
            bsTooltip("enable_turnover", "Enable daily discharges/admissions (dynamic census).", placement = "right", trigger = "hover", options = list(container = "body")),
            bsTooltip("daily_discharge_frac", "Fraction of patients discharged per ward per day.", placement = "right", trigger = "hover", options = list(container = "body")),
            bsTooltip("daily_discharge_min_per_ward", "Minimum discharges per ward per day when turnover enabled.", placement = "right", trigger = "hover", options = list(container = "body")),
            bsTooltip("p_admit_import_cs", "P(new admission is CS on arrival).", placement = "right", trigger = "hover", options = list(container = "body")),
            bsTooltip("p_admit_import_cr", "P(new admission is CR on arrival).", placement = "right", trigger = "hover", options = list(container = "body")),
            
            bsTooltip("enable_admit_import_seasonality", "Enable time-varying admission importation probabilities (requires turnover).", placement = "right", trigger = "hover", options = list(container = "body")),
            bsTooltip("admit_import_seasonality", "Choose sinusoid (smooth), piecewise (high/low), or shock (single random pulse).", placement = "right", trigger = "hover", options = list(container = "body")),
            bsTooltip("admit_import_period_days", "Season length in days (used by sinusoid/piecewise; shock ignores period).", placement = "right", trigger = "hover", options = list(container = "body")),
            bsTooltip("admit_import_amp", "Sinusoid amplitude. 0.5 means ±50% around baseline.", placement = "right", trigger = "hover", options = list(container = "body")),
            bsTooltip("admit_import_phase_day", "Shifts the day within the period when the sinusoid phase is anchored.", placement = "right", trigger = "hover", options = list(container = "body")),
            bsTooltip("admit_import_high_start_day", "Piecewise high-season start day within the period.", placement = "right", trigger = "hover", options = list(container = "body")),
            bsTooltip("admit_import_high_end_day", "Piecewise high-season end day within the period.", placement = "right", trigger = "hover", options = list(container = "body")),
            bsTooltip("admit_import_high_mult", "Multiplier applied during high season.", placement = "right", trigger = "hover", options = list(container = "body")),
            bsTooltip("admit_import_low_mult", "Multiplier applied outside high season.", placement = "right", trigger = "hover", options = list(container = "body")),
            bsTooltip("admit_import_pmax_cs", "Cap for CS import probability after scaling.", placement = "right", trigger = "hover", options = list(container = "body")),
            bsTooltip("admit_import_pmax_cr", "Cap for CR import probability after scaling.", placement = "right", trigger = "hover", options = list(container = "body")),
            bsTooltip("admit_import_shock_min_days", "Shock mode: minimum duration (days).", placement = "right", trigger = "hover", options = list(container = "body")),
            bsTooltip("admit_import_shock_max_days", "Shock mode: maximum duration (days). Clipped to <= num_days.", placement = "right", trigger = "hover", options = list(container = "body")),
            bsTooltip("admit_import_shock_mult_min", "Shock mode: multiplier lower bound.", placement = "right", trigger = "hover", options = list(container = "body")),
            bsTooltip("admit_import_shock_mult_max", "Shock mode: multiplier upper bound.", placement = "right", trigger = "hover", options = list(container = "body")),
            
            bsTooltip("enable_screening_controls", "Enable screening schedule/delay/observation controls.", placement = "right", trigger = "hover", options = list(container = "body")),
            bsTooltip("screen_every_k_days", "Screen the hospital every k days (k=7 weekly).", placement = "right", trigger = "hover", options = list(container = "body")),
            bsTooltip("screen_result_delay_days", "Lab delay (days) before results are available.", placement = "right", trigger = "hover", options = list(container = "body")),
            bsTooltip("screen_on_admission", "If enabled, screen patients at admission (subject to delay).", placement = "right", trigger = "hover", options = list(container = "body")),
            bsTooltip("persist_observations", "If enabled, observed screening status persists across days.", placement = "right", trigger = "hover", options = list(container = "body")),
            
            bsTooltip("override_staff_wards_per_staff", "Override staff multi-ward assignment to control cross-ward coupling.", placement = "right", trigger = "hover", options = list(container = "body")),
            bsTooltip("staff_wards_per_staff", "How many wards each staff member covers (higher => more coupling).", placement = "right", trigger = "hover", options = list(container = "body")),
            
            bsTooltip("dataset", "Generate learn first (writes early-warning threshold JSON), then test reuses it.", placement = "right", trigger = "hover", options = list(container = "body")),
            bsTooltip("label_horizons", "Horizons (days ahead) for which convert_to_pt.py computes labels (e.g., 7,14,21,30).", placement = "right", trigger = "hover", options = list(container = "body")),
            bsTooltip("python_bin", "Python executable used to run generate_amr_data.py and convert_to_pt.py.", placement = "right", trigger = "hover", options = list(container = "body")),
            bsTooltip("export_gif", "If unchecked, passes --no_export_gif to avoid Pillow dependency.", placement = "right", trigger = "hover", options = list(container = "body")),
            bsTooltip("script_dir", "Working directory containing generate_amr_data.py and convert_to_pt.py.", placement = "right", trigger = "hover", options = list(container = "body")),
            bsTooltip("state_mode", "ground_truth uses latent amr_state; partial_observation uses observed-positive status in the converter.", placement = "right", trigger = "hover", options = list(container = "body")),
            bsTooltip("conv_workers", "Number of parallel workers for convert_to_pt.py. Use 0 for serial conversion.", placement = "right", trigger = "hover", options = list(container = "body")),
            bsTooltip("keep_graphml", "If checked, passes --keep_graphml so GraphML files are retained after PT conversion.", placement = "right", trigger = "hover", options = list(container = "body")),
            bsTooltip("use_pt_out_dir", "If checked, also passes --pt_out_dir to archive generated PT files in a separate folder.", placement = "right", trigger = "hover", options = list(container = "body")),
            bsTooltip("pt_out_dir", "Optional folder used by convert_to_pt.py to copy generated PT files in addition to the main output location.", placement = "right", trigger = "hover", options = list(container = "body")),
            bsTooltip("run_btn", "Runs simulation then converts GraphML → PT tensors.", placement = "right", trigger = "hover", options = list(container = "body"))
          )
        )
      ),
      
      tabItem(
        tabName = "logs",
        fluidRow(
          box(
            title = "Status",
            width = 12,
            status = "warning",
            solidHeader = TRUE,
            verbatimTextOutput("log")
          )
        )
      )
    )
  )
)

server <- function(input, output, session) {
  
  log_text <- reactiveVal("")
  output$log <- renderText(log_text())
  
  # Keep dropdown staff list valid as num_staff changes
  observeEvent(input$num_staff, {
    n <- max(1, as.integer(input$num_staff))
    choices <- paste0("s", 0:(n - 1))
    updateSelectInput(session, "superspreader_staff_dropdown", choices = choices, selected = choices[1])
  }, ignoreInit = FALSE)
  
  # Resolve superspreader staff ID according to mode (reproducible)
  resolved_superspreader <- reactive({
    if (!isTRUE(input$enable_superspreader)) return("")
    
    mode <- as.character(input$superspreader_mode)
    n_staff <- max(1, as.integer(input$num_staff))
    
    if (mode == "manual") {
      s <- trimws(as.character(input$superspreader_staff_manual))
      return(s)
    }
    
    if (mode == "dropdown") {
      return(as.character(input$superspreader_staff_dropdown))
    }
    
    # mode == "random"
    set.seed(as.integer(input$seed))
    k <- sample.int(n_staff, size = 1) - 1
    return(paste0("s", k))
  })
  
  output$superspreader_resolved <- renderText({
    s <- resolved_superspreader()
    if (!isTRUE(input$enable_superspreader)) {
      return("Superspreader: (disabled)")
    }
    if (!nzchar(s)) {
      return("Superspreader: (no staff selected)")
    }
    paste0("Superspreader (resolved): ", s)
  })
  
  observeEvent(input$run_btn, {
    
    shinyjs::disable("run_btn")
    log_text("Running AMR simulation...\n")
    
    # --- set working directory exactly like the working app ---
    script_dir <- normalizePath(input$script_dir, winslash = "/", mustWork = TRUE)
    old_wd <- getwd()
    setwd(script_dir)
    on.exit(setwd(old_wd), add = TRUE)
    
    # --- output folder ---
    output_dir <- paste0("synthetic_amr_graphs_", input$dataset)
    
    # --- run simulator ---
    sim_args <- c(
      "generate_amr_data.py",
      "--output_dir", output_dir,
      "--num_regions", as.character(input$num_regions),
      "--seed", as.character(input$seed),
      "--num_days", as.character(input$num_days),
      "--num_patients", as.character(input$num_patients),
      "--num_staff", as.character(input$num_staff),
      "--num_wards", as.character(input$num_wards),
      "--export_yaml"
    )
    
    # GIF toggle
    if (!isTRUE(input$export_gif)) {
      sim_args <- c(sim_args, "--no_export_gif")
    }
    
    # Superspreader (optional)
    ss <- resolved_superspreader()
    if (isTRUE(input$enable_superspreader) && nzchar(ss)) {
      sim_args <- c(
        sim_args,
        "--superspreader_staff", as.character(ss),
        "--superspreader_state", as.character(input$superspreader_state),
        "--superspreader_start_day", as.character(input$superspreader_start_day),
        "--superspreader_end_day", as.character(input$superspreader_end_day),
        "--superspreader_patient_frac_mult", as.character(input$superspreader_patient_frac_mult),
        "--superspreader_patient_min_add", as.character(input$superspreader_patient_min_add),
        "--superspreader_staff_contacts", as.character(input$superspreader_staff_contacts),
        "--superspreader_edge_weight_mult", as.character(input$superspreader_edge_weight_mult)
      )
    }
    
    # Turnover (optional)
    if (isTRUE(input$enable_turnover)) {
      sim_args <- c(
        sim_args,
        "--daily_discharge_frac", as.character(input$daily_discharge_frac),
        "--daily_discharge_min_per_ward", as.character(input$daily_discharge_min_per_ward),
        "--p_admit_import_cs", as.character(input$p_admit_import_cs),
        "--p_admit_import_cr", as.character(input$p_admit_import_cr)
      )
    }
    
    # Seasonal admission importation (optional; gated)
    if (isTRUE(input$enable_turnover) &&
        isTRUE(input$enable_admit_import_seasonality) &&
        as.character(input$admit_import_seasonality) != "none") {
      
      mode <- as.character(input$admit_import_seasonality)
      
      # Always pass mode + period + caps (period is ignored by shock but harmless to pass)
      sim_args <- c(
        sim_args,
        "--admit_import_seasonality", mode,
        "--admit_import_period_days", as.character(input$admit_import_period_days),
        "--admit_import_pmax_cs", as.character(input$admit_import_pmax_cs),
        "--admit_import_pmax_cr", as.character(input$admit_import_pmax_cr)
      )
      
      if (mode == "sinusoid") {
        sim_args <- c(
          sim_args,
          "--admit_import_amp", as.character(input$admit_import_amp),
          "--admit_import_phase_day", as.character(input$admit_import_phase_day)
        )
      }
      
      if (mode == "piecewise") {
        sim_args <- c(
          sim_args,
          "--admit_import_high_start_day", as.character(input$admit_import_high_start_day),
          "--admit_import_high_end_day", as.character(input$admit_import_high_end_day),
          "--admit_import_high_mult", as.character(input$admit_import_high_mult),
          "--admit_import_low_mult", as.character(input$admit_import_low_mult)
        )
      }
      
      if (mode == "shock") {
        sim_args <- c(
          sim_args,
          "--admit_import_shock_min_days", as.character(input$admit_import_shock_min_days),
          "--admit_import_shock_max_days", as.character(input$admit_import_shock_max_days),
          "--admit_import_shock_mult_min", as.character(input$admit_import_shock_mult_min),
          "--admit_import_shock_mult_max", as.character(input$admit_import_shock_mult_max)
        )
      }
    }
    
    # Screening controls (optional)
    if (isTRUE(input$enable_screening_controls)) {
      sim_args <- c(
        sim_args,
        "--screen_every_k_days", as.character(input$screen_every_k_days),
        "--screen_result_delay_days", as.character(input$screen_result_delay_days)
      )
      
      sim_args <- c(sim_args, "--screen_on_admission", ifelse(isTRUE(input$screen_on_admission), "1", "0"))
      
      if (isTRUE(input$persist_observations)) {
        sim_args <- c(sim_args, "--persist_observations", "1")
      } else {
        sim_args <- c(sim_args, "--persist_observations", "0")
      }
    }
    
    # Cross-ward mixing override (optional)
    if (isTRUE(input$override_staff_wards_per_staff)) {
      sim_args <- c(
        sim_args,
        "--staff_wards_per_staff", as.character(input$staff_wards_per_staff)
      )
    }
    
    log_text(
      paste0(
        log_text(),
        "Command:\n", as.character(input$python_bin), " ",
        paste(sim_args, collapse = " "),
        "\n\n"
      )
    )
    
    # Clean output directory before simulation
    if (dir.exists(output_dir)) {
      graphml_files <- list.files(
        output_dir,
        pattern = "\\.graphml$",
        full.names = TRUE
      )
      
      if (length(graphml_files) > 0) {
        message(sprintf(
          "DT_SIM_CLEAN removing %d stale GraphML files from %s",
          length(graphml_files),
          output_dir
        ))
        file.remove(graphml_files)
      }
    }
    
    system2(as.character(input$python_bin), args = sim_args)
    
    log_text(
      paste0(
        log_text(),
        "\nSimulation completed.\n\n"
      )
    )
    
    # GraphML → PT conversion
    threshold_json <- file.path("synthetic_amr_graphs_learn", "labels", "early_warning_threshold.json")
    label_dir <- file.path(output_dir, "labels")

    if (dir.exists(output_dir)) {
      pt_files <- list.files(
        output_dir,
        pattern = "\\.pt$",
        full.names = TRUE
      )

      if (length(pt_files) > 0) {
        message(sprintf(
          "DT_CONV_CLEAN removing %d stale PT files from %s",
          length(pt_files),
          output_dir
        ))
        file.remove(pt_files)
      }
    }

    if (dir.exists(label_dir)) {
      unlink(label_dir, recursive = TRUE, force = TRUE)
    }

    conv_args <- c(
      "convert_to_pt.py",
      "--graphml_dir", output_dir,
      "--horizons", as.character(input$label_horizons),
      "--state_mode", as.character(input$state_mode),
      "--workers", as.character(max(0L, as.integer(input$conv_workers)))
    )

    if (isTRUE(input$keep_graphml)) {
      conv_args <- c(conv_args, "--keep_graphml")
    }

    if (isTRUE(input$use_pt_out_dir)) {
      pt_out_dir <- trimws(as.character(input$pt_out_dir))
      if (nzchar(pt_out_dir)) {
        conv_args <- c(conv_args, "--pt_out_dir", pt_out_dir)
      } else {
        log_text(
          paste0(
            log_text(),
            "WARNING: PT archive folder is empty, so --pt_out_dir was not passed.\n\n"
          )
        )
      }
    }

    if (input$dataset == "learn") {
      conv_args <- c(
        conv_args,
        "--early_res_frac_threshold_out", threshold_json
      )
    } else {
      if (!file.exists(threshold_json)) {
        log_text(
          paste0(
            log_text(),
            "ERROR: early-warning threshold file not found:\n  ",
            threshold_json,
            "\n\nRun a 'learn' simulation first to create it.\n"
          )
        )
        shinyjs::enable("run_btn")
        return(NULL)
      }

      conv_args <- c(
        conv_args,
        "--early_res_frac_threshold_file", threshold_json
      )
    }

    log_text(
      paste0(
        log_text(),
        "Converting GraphML to PT:\n", as.character(input$python_bin), " ",
        paste(conv_args, collapse = " "),
        "\n\n"
      )
    )

    system2(as.character(input$python_bin), args = conv_args)

    log_text(
      paste0(
        log_text(),
        "\nConversion completed.\n"
      )
    )
    
    shinyjs::enable("run_btn")
  })
}

shinyApp(ui, server)