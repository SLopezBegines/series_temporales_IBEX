# Creat directories to store data
create_directories <- function(base_path) {
  # Helper function to create a directory if it does not exist
  create_dir_if_not_exists <- function(path) {
    if (!dir.exists(path)) {
      dir.create(path, recursive = TRUE, showWarnings = FALSE)
      cat("Directory created:", path, "\n")
    } else {
      cat("Directory already exists:", path, "\n")
    }
  }

  # Create the base directory and its subdirectories
  create_dir_if_not_exists(base_path)
  create_dir_if_not_exists(paste0(base_path, "/tables"))
  create_dir_if_not_exists(paste0(base_path, "/figures"))
  create_dir_if_not_exists(paste0(base_path, "/RData"))
  create_dir_if_not_exists(paste0(base_path, "/log"))
}
# Example usage:
# create_directories("output/")


# Save Plots

# FUNCIÓN AUXILIAR: GUARDAR GRÁFICOS EN MÚLTIPLES FORMATOS


save_plot <- function(plotname,
                      plot,
                      output_dir = output_path,
                      image_number = NULL,
                      width = NULL,
                      height = NULL,
                      dpi = 300,
                      formats = c("tiff", "pdf"),
                      print_plot = TRUE,
                      cleanup = FALSE) {
  #' Guarda gráficos ggplot2 en múltiples formatos
  #'
  #' @param plotname Nombre base del archivo (sin extensión)
  #' @param plot Objeto ggplot2
  #' @param output_dir Directorio de salida
  #' @param image_number Número de imagen para prefijo (opcional)
  #' @param width Ancho en cm
  #' @param height Alto en cm
  #' @param dpi Resolución para formatos raster
  #' @param formats Vector de formatos ("tiff", "pdf", "png")
  #' @param print_plot Imprimir plot en consola
  #' @param cleanup Eliminar objeto plot después de guardar
  #' @return Ruta del primer archivo guardado (invisible)

  # Verificar objeto ggplot
  if (!inherits(plot, "ggplot")) {
    warning(paste("El objeto", plotname, "no es un ggplot válido. Omitiendo."))
    return(invisible(NULL))
  }

  # Crear directorio si no existe
  figures_dir <- file.path(output_dir, "/figures")
  if (!dir.exists(figures_dir)) {
    dir.create(figures_dir, recursive = TRUE)
  }

  # Construir nombre base del archivo
  # Usar variable global si no se especifica
  if (is.null(image_number)) {
    image_number <- get("image_number", envir = .GlobalEnv)
  }

  base_filename <- sprintf("%03d_%s", image_number, plotname)


  # Guardar en cada formato
  saved_paths <- c()

  for (format in formats) {
    extension <- paste0(".", format)
    filepath <- file.path(figures_dir, paste0(base_filename, extension))

    tryCatch(
      {
        # Construir argumentos condicionalmente
        save_args <- list(
          filename = filepath,
          plot = plot,
          units = "cm"
        )
        # Agregar dimensiones solo si se especifican
        if (!is.null(width)) save_args$width <- width
        if (!is.null(height)) save_args$height <- height

        # Solo agregar dpi para formatos raster
        if (format %in% c("tiff", "png")) {
          save_args$dpi <- dpi
        }

        do.call(ggsave, save_args)
        saved_paths <- c(saved_paths, filepath)
      },
      error = function(e) {
        warning(paste("No se pudo guardar:", filepath, "- Error:", e$message))
      }
    )
  }

  # Incrementar contador global
  assign("image_number", image_number + 1, envir = .GlobalEnv)
  # Imprimir plot si se solicita
  if (print_plot) {
    print(plot)
  }

  # Limpiar objeto si se solicita
  if (cleanup && exists("plot", inherits = FALSE)) {
    rm(plot, envir = parent.frame())
  }

  return(invisible(saved_paths[1]))
}


# Función mejorada para guardar gráficos base R con soporte para múltiples plots
save_base_plot <- function(plotname,
                           plot_function,
                           show_plot = TRUE,
                           par_settings = NULL) {
  filename <- file.path(output_path, "/figures", paste0(sprintf("%03d", image_number), "_", plotname))

  tryCatch(
    {
      # Guardar TIFF
      tiff(file = paste0(filename, ".tiff"))
      if (!is.null(par_settings)) {
        do.call(par, par_settings)
      }
      plot_function()
      dev.off()

      # Guardar PDF
      pdf(file = paste0(filename, ".pdf"))
      if (!is.null(par_settings)) {
        do.call(par, par_settings)
      }
      plot_function()
      dev.off()

      # Mostrar en pantalla si se desea
      if (show_plot) {
        if (!is.null(par_settings)) {
          do.call(par, par_settings)
        }
        plot_function()
        p <- recordPlot()
      }

      image_number <<- image_number + 1

      # Devolver el plot capturado
      if (show_plot) {
        return(invisible(p))
      }
    },
    error = function(e) {
      warning(paste("No se pudo guardar la imagen base R:", filename, "Error:", e$message))
    }
  )
}
## Ejemplo de uso:
# 4. O todos en un grid 3x2
# save_base_plot(
#   plotname = "acf_pacf_completo",
#   plot_function = function() {
#     acf(ibex_close_xts, lag.max = 40, main = "ACF - Cierre")
#     pacf(ibex_close_xts, lag.max = 40, main = "PACF - Cierre")
#     acf(ibex_return_xts, lag.max = 40, main = "ACF - Retornos")
#     pacf(ibex_return_xts, lag.max = 40, main = "PACF - Retornos")
#     returns_sq <- ibex_return_xts^2
#     acf(returns_sq, lag.max = 40, main = "ACF - Retornos²")
#     pacf(returns_sq, lag.max = 40, main = "PACF - Retornos²")
#   },
#   par_settings = list(mfrow = c(3, 2)),
#   width = 12,
#   height = 10
# )

## reactViewTable:
### librería: reactable
### objetivo: ver los datos de forma interactiva y dinámica en HTML
reactViewTable <- function(data) {
  reactable::reactable(
    data,
    bordered = TRUE,
    borderless = FALSE,
    highlight = TRUE,
    outlined = TRUE,
    resizable = TRUE,
    filterable = TRUE,
    searchable = TRUE,
    showSortIcon = TRUE,
    showSortable = TRUE,
    showPageSizeOptions = TRUE,
    defaultPageSize = 15,
    pageSizeOptions = c(5, 10, 20, 50, 100),
    width = "100%",
    theme = reactable::reactableTheme(
      headerStyle = list(
        backgroundColor = "#095540",
        color = "white",
        fontWeight = "bold"
      ),
      rowStyle = list(
        backgroundColor = "#efeee0"
      ),
      borderColor = "#ccc",
      stripedColor = "#f5f5f5",
      highlightColor = "#e0f7e9"
    ),
    defaultColDef = reactable::colDef(
      format = reactable::colFormat(digits = 2),
      align = "center"
    )
  )
}


## Función visualización derivadas: línea e histograma ####
diff_plot <- function(df, variables, labels = NULL, width = NULL, height = NULL) {
  # Validar que variables es un vector
  if (!is.character(variables)) {
    stop("El parámetro 'variables' debe ser un vector de caracteres")
  }

  # Si no se proporcionan labels, usar nombres de variables
  if (is.null(labels)) {
    labels <- variables
  }

  # Validar que labels tiene la misma longitud que variables
  if (length(labels) != length(variables)) {
    stop("La longitud de 'labels' debe coincidir con la longitud de 'variables'")
  }

  # Crear lista para almacenar gráficos
  plot_list <- list()

  # Generar gráficos para cada variable
  for (i in seq_along(variables)) {
    variable <- variables[i]
    label <- labels[i]

    # Conversión para evaluar el string como columna
    var_sym <- rlang::sym(variable)

    # Verificar que la variable existe
    if (!variable %in% names(df)) {
      warning(paste("Variable", variable, "no existe en el dataframe. Omitiendo."))
      next
    }

    # Gráfico de línea temporal
    p_line <- df %>%
      ggplot(aes(x = date, y = !!var_sym)) +
      geom_line(color = "#E74C3C") +
      facet_wrap(~index, scales = "free_y") +
      labs(
        title = paste("Serie temporal:", label),
        y = label,
        x = "Fecha"
      ) +
      theme_minimal(base_size = 10) +
      theme(
        plot.title = element_text(size = 10),
        plot.subtitle = element_text(size = 8)
      )

    # Filtrar NAs para el histograma
    df_clean <- df %>% filter(!is.na(!!var_sym))

    # Verificar que hay datos después de filtrar NAs
    if (nrow(df_clean) == 0) {
      warning(paste("Variable", variable, "solo contiene NAs. Omitiendo."))
      next
    }

    # Estadísticos para la normal teórica
    var_mean <- mean(df_clean[[variable]], na.rm = TRUE)
    var_sd <- sd(df_clean[[variable]], na.rm = TRUE)

    # Histograma + densidades
    p_hist <- ggplot(df_clean, aes(x = !!var_sym)) +
      geom_histogram(aes(y = after_stat(density)),
        bins = 50,
        fill = "#3498DB",
        alpha = 0.7,
        color = "white"
      ) +
      geom_density(color = "#E74C3C", linewidth = 0.5) +
      stat_function(
        fun = dnorm,
        args = list(mean = var_mean, sd = var_sd),
        color = "#27AE60",
        linewidth = 0.7,
        linetype = "dashed"
      ) +
      labs(
        title = paste("Distribución:", label),
        subtitle = "Rojo: Densidad empírica | Verde: Normal teórica",
        x = label,
        y = "Densidad"
      ) +
      theme_minimal(base_size = 10) +
      theme(
        plot.title = element_text(size = 10),
        plot.subtitle = element_text(size = 8),
        axis.text = element_text(size = 8),
        axis.title = element_text(size = 8)
      )

    # Agregar ambos gráficos a la lista
    plot_list <- c(plot_list, list(p_line, p_hist))
  }

  # Combinar todos los gráficos
  combined_plot <- wrap_plots(plot_list, ncol = 2)
  print(combined_plot)
  save_plot("distribucion_diferencias", combined_plot)
  return(combined_plot)
}

# Log_cat function
# Captura todos los "cat()" y los guarda en un archivo de log e imprime en la consola.

log_cat <- function(...) {
  msg <- paste0(...)
  cat(msg) # Imprime en consola
  cat(msg, file = log_file, append = TRUE) # Guarda en archivo
}

# Reemplaza todos los cat() por log_cat()
# log_cat("=== CONFIGURACIÓN ===\n")
# log_cat("Fecha inicio:", format(start_date, "%Y-%m-%d"), "\n")
# etc...
# Ejemplo de uso:
# Al inicio del script
# log_file <- file.path("data", paste0("download_log_", format(Sys.time(), "%Y%m%d_%H%M%S"), ".txt"))


# Función para guardar tablas

save_table <- function(name, data, output_dir = paste0(output_path, "/tables")) {
  # Verificar que table_number existe en el ambiente global
  if (!exists("table_number", envir = .GlobalEnv)) {
    stop("La variable global 'table_number' no existe")
  }

  # Crear directorio si no existe
  if (!dir.exists(output_dir)) {
    dir.create(output_dir, recursive = TRUE)
  }

  # Formatear nombre de archivo
  file_name <- sprintf("%02d_%s.xlsx", table_number, name)
  file_path <- file.path(output_dir, file_name)

  # Guardar archivo
  writexl::write_xlsx(data, file_path)

  # Incrementar table_number globalmente
  assign("table_number", table_number + 1, envir = .GlobalEnv)

  # Mensaje de confirmación
  message(sprintf("Tabla guardada: %s", file_path))

  invisible(file_path)
}

# Uso
# save_table("table", df)  # Guarda en output/tables/01_table.xlsx
# save_table("summary", df2)  # Guarda en output/tables/02_summary.xlsx
# save_table("results", df3, "output/results")  # Guarda en output/results/03_results.xlsx


# Función para detectar formato automáticamente e impromir tablas en formato html o pdf
tabla_auto <- function(df, caption) {
  # Escapa caracteres especiales de LaTeX en caption
  caption_safe <- gsub("&", "\\\\&", caption)
  if (knitr::is_html_output()) {
    df %>%
      kable(format = "html", escape = FALSE, caption = caption) %>%
      kable_styling(
        bootstrap_options = c("striped", "hover", "condensed"),
        full_width = TRUE, position = "left"
      ) %>%
      column_spec(2:3, width = "25%")
  } else if (knitr::is_latex_output()) {
    df %>%
      kable(format = "latex", escape = FALSE, booktabs = TRUE, caption = caption_safe) %>%
      kable_styling(
        latex_options = c("scale_down", "HOLD_position"),
        full_width = TRUE
      )
  } else {
    kable(df, caption = caption)
  }
}
