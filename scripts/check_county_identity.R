# Independent base-R check of published geographic codes and descriptive counts.
args <- commandArgs(trailingOnly = TRUE)
root <- if (length(args)) args[[1]] else "."
read_report <- function(name) {
  read.csv(file.path(root, "reports", name), colClasses = "character", check.names = FALSE)
}
crosswalk <- read_report("county-identity-crosswalk.csv")
summary <- read_report("county-identity-summary.csv")
stopifnot(nrow(crosswalk) > 0, !anyDuplicated(crosswalk$COUNTY_ID))
stopifnot(all(crosswalk$status %in% c("reference_match", "unmatched", "ambiguous_reference")))
matched <- crosswalk[crosswalk$status == "reference_match", ]
stopifnot(all(grepl("^[0-9]{2}$", matched$state_ansi)))
stopifnot(all(grepl("^[0-9]{3}$", matched$county_ansi)))
stopifnot(all(grepl("^[0-9]{5}$", matched$fips)), !anyDuplicated(matched$fips))
stopifnot(all(matched$fips == paste0(matched$state_ansi, matched$county_ansi)))
stopifnot(!anyDuplicated(summary$state), setequal(summary$state, unique(crosswalk$state)))
for (i in seq_len(nrow(summary))) {
  state <- summary$state[[i]]
  stopifnot(sum(crosswalk$state == state) == as.integer(summary$training_counties[[i]]))
  stopifnot(sum(matched$state == state) == as.integer(summary$matched_counties[[i]]))
}
print(summary, row.names = FALSE)
cat("Verified ", nrow(crosswalk), " historical county identities using ", R.version.string,
    "; no target values or model fitting.\n", sep = "")
