library(devtools)
#devtools::install_github('JGCRI/gcamdata')
library(gcamdata)
library(dplyr)

# ------------------------------------------------------------------------
# Write XML File
# ------------------------------------------------------------------------

setwd('/cluster/tufts/lamontagnelab/abirnb01/GCIMS/Abby_paper/stochastic_runoff_csv_worst')

#for (i in 1:100) {
for (i in 1:50) {

data <- read.csv(paste('stochastic_runoff_',as.character(i),'.csv',sep=''))
colnames(data) <- gsub('X','',colnames(data))

out_xml_name <- paste('stochastic_runoff_',as.character(i),'.xml',sep='')
output_xml_dir <- paste0('/cluster/tufts/lamontagnelab/abirnb01/GCIMS/Abby_paper/stochastic_runoff_xml_worst')
gcam_file <- file.path(output_xml_dir,out_xml_name)

gcamdata::create_xml(gcam_file) %>%
  gcamdata::add_xml_data(data, "GrdRenewRsrcMaxNoFillOut")%>%
  gcamdata::run_xml_conversion()
}
