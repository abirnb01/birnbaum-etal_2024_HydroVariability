args = base::commandArgs(trailingOnly=TRUE) #for running as job array on cluster

#load in required libraries
library(rgcam)
library(dplyr)

queryName<-'ag_production_allbasin' #specify query that we want to extract

#get list of all GCAM databases
dbpath1<-"/cluster/tufts/lamontagnelab/abirnb01/GCIMS/Abby_paper/GCAM_output/version2/"
dbs1<-list.dirs(dbpath1,recursive = FALSE)
dbs<-c(dbs1)

#define function to make query
make_query<-function(scenario){
dbLoc<-paste0(scenario)
queryFile<-paste0('/cluster/tufts/lamontagnelab/abirnb01/GCIMS/Abby_paper/queries/query_xml/',queryName,'.xml')
queryData=paste0('/cluster/tufts/lamontagnelab/abirnb01/GCIMS/Abby_paper/queries/temp_data_files/',queryName,"_",as.character(args[1]),'.dat')
queryResult<-rgcam::addScenario(dbLoc,queryData,queryFile=queryFile)
file.remove(queryData)
return(queryResult[[1]][[1]])
}

#run query for all databases
make_query(dbs[as.numeric(args[1])]) %>%
  readr::write_csv(paste0('/cluster/tufts/lamontagnelab/abirnb01/GCIMS/Abby_paper/queries/query_results/',queryName,'/',queryName,'_',as.numeric(args[1]),'.csv'))