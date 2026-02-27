# Load Python results into R
stocks <- py_to_r(py$results_indexes)

# Remove stocks with zero rows
stocks <- stocks[sapply(stocks, function(x) nrow(x) > 0)]

stocks <- Map(function(df, name) {
  df$date <- as.Date(row.names(df))
  df$index <- name
  colnames(df) <- c("close", "high", "low", "open", "volume", "date", "index")
  return(df)
}, stocks, names(stocks))

# Save Stocks list
saveRDS(stocks, paste0(output_path, "/RData/stocks_list.rds"))

# All dfs bind rows
all_stocks_df <- bind_rows(stocks, .id = "index")
all_stocks_df <- all_stocks_df |>
  dplyr::mutate(
    index = as.factor(index),
    date = as.Date(date)
  )

# Load Python results into R
stocks_companies <- py_to_r(py$results_companies)
# Remove companies with zero rows
stocks_companies <- stocks_companies[sapply(stocks_companies, function(x) nrow(x) > 0)]

stocks_companies <- Map(function(df, name) {
  df$date <- as.Date(row.names(df))
  # df$name <- name
  colnames(df) <- c("close", "high", "low", "open", "volume", "ticker", "name", "sector", "date")
  return(df)
}, stocks_companies, names(stocks_companies))

# Save Stocks list
saveRDS(stocks_companies, paste0(output_path, "/RData/stocks_companies.rds"))

# All dfs bind rows
all_companies_df <- bind_rows(stocks_companies, .id = "index")

all_companies_df <- all_companies_df |>
  dplyr::mutate(
    index = as.factor(index),
    ticker = as.factor(ticker),
    name = as.factor(name),
    sector = as.factor(sector),
    date = as.Date(date)
  )

# Save data as parquet
writexl::write_xlsx(stocks, paste0(output_path, "/tables/stocks.xlsx"))
writexl::write_xlsx(stocks_companies, paste0(output_path, "/tables/stocks_companies.xlsx"))

write_parquet(all_stocks_df, paste0(output_path, "/tables/all_indices.parquet"))
write_parquet(all_companies_df, paste0(output_path, "/tables/ibex35_companies_all.parquet"))
rm(stocks, stocks_companies)
