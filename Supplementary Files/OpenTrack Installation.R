# OpenTrack Installation Script

# This script is designed to install OpenTrack from a brand new R/Rstudio
# install. After this script has finished, navigate to the 'OpenTrack' 
# project in the 'R/OpenTrack' folder, and run the command: 
# OpenTrack::run_app()

# -------------------------------------------------------------------------

# find the filepath to R
install.packages("stringr", type = "binary")
r.path <- paste0(stringr::str_split(getwd(), "/R/")[[1]][1], "/R/")

# set the R folder as the wd
setwd(r.path)

# download a copy of the repo
utils::download.file(
    url = "https://github.com/joseph-shaw/OpenTrack/archive/refs/heads/main.zip",
    destfile = "OpenTrack.zip"
    )

# unzip the .zip file
utils::unzip(zipfile = "OpenTrack.zip")

# rename
file.rename("OpenTrack-main", "OpenTrack")

# set OpenTrack as the wd
setwd(paste0(r.path, "OpenTrack"))

# install devtools
install.packages("devtools")

# install rtools
install.packages("installr")
installr::install.rtools()

# install specific version of shinydashboardPlus
devtools::install_version(
    "shinydashboardPlus", 
    version = "0.7.5", 
    repos = "http://cran.us.r-project.org", 
    type = "source")

# install other dependencies
install.packages(
   c('config', 'golem', 'gdtools', 'attempt', 'DT', 'dplyr', 'DBI', 
     'DescTools', 'caTools', 'chron', 'dashboardthemes', 'data.table',
     'dygraphs', 'extrafont', 'fmsb', 'ggiraph', 'ggplot2', 'gridExtra',
     'lubridate', 'odbc', 'plotly', 'png', 'pool', 'rowr', 'shinyFiles',
     'shinyWidgets', 'shinycssloaders', 'shinyjs', 'tidyr', 'xts', 'zoo',
     'formattable', 'signal'),
   type = "binary"
)
devtools::install_github("cvarrichio/rowr") 

# install OpenTrack
#devtools::install_github("joseph-shaw/OpenTrack")
install.packages(paste0(r.path, "OpenTrack"), 
                 repos = NULL, 
                 type = "source", dependencies = T)


# -------------------------------------------------------------------------
# At this point, navigate to the 'OpenTrack' project in the 'R/OpenTrack' 
# folder, and run the command: OpenTrack::run_app()
