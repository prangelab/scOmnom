## ----echo = FALSE, message = FALSE--------------------------------------------
library(knitr)
knitr::opts_chunk$set(
    tidy  = FALSE,
    comment = NA,
    fig.align = "center")
library(GetoptLong)

## ----main, eval = FALSE-------------------------------------------------------
#  library(GetoptLong)
#  
#  subCommands(
#  	"sub1", "sub1.R",
#  	        "This is the description of sub command 1, which will be a long long long text.",
#  	"sub2", "sub2.R",
#  	        "This is the description of sub command 2, which will be a long long long text."
#  )

## ----sub1, eval = FALSE-------------------------------------------------------
#  library(GetoptLong)
#  
#  GetoptLong("foo=i", "Set value for foo.")
#  
#  qqcat("foo is @{foo}\n")

## ----sub2, eval = FALSE-------------------------------------------------------
#  library(GetoptLong)
#  
#  GetoptLong("bar=i", "Set value for bar")
#  
#  qqcat("bar is @{bar}\n")

## ----echo = FALSE, eval = TRUE, results = "asis"------------------------------
chunks <- knitr:::knit_code$get()
writeLines(chunks[["main"]], con = "main.R")
writeLines(chunks[["sub1"]], con = "sub1.R")
writeLines(chunks[["sub2"]], con = "sub2.R")
cat("```\n")
cat(system("Rscript main.R", intern = TRUE), sep = "\n")
cat("```\n")

## ----echo = FALSE, eval = TRUE, results = "asis"------------------------------
cat("```\n")
cat(system("Rscript main.R sub1 --help", intern = TRUE), sep = "\n")
cat("```\n")

## ----echo = FALSE, eval = TRUE, results = "asis"------------------------------
cat("```\n")
cat(system("Rscript main.R sub1 --foo 10", intern = TRUE), sep = "\n")
cat("```\n")

## ----eval = FALSE-------------------------------------------------------------
#  subCommands(
#  	"sub1", "path/sub1.R", "description 1",
#  	"sub2", "path/sub2.R", "description 2"
#  )

## ----eval = FALSE-------------------------------------------------------------
#  subCommands(
#  	"path/sub1.R", "description 1",
#  	"path/sub2.R", "description 2"
#  )

## ----section, eval = FALSE----------------------------------------------------
#  subCommands(
#  	"----", "----", "This is section1",
#  	"sub1", "sub1.R",
#  	      "This is the description of sub command 1, which will be a long long long text.",
#  
#  	"----", "----", "This is section2",
#  	"sub2", "sub2.R",
#  	      "This is the description of sub command 2, which will be a long long long text."
#  )

## ----echo = FALSE, eval = TRUE, results = "asis"------------------------------
chunks <- knitr:::knit_code$get()
writeLines(c("library(GetoptLong)", chunks[["section"]]), con = "main.R")

cat("```\n")
cat(system("Rscript main.R", intern = TRUE), sep = "\n")
cat("```\n")

## ----head-foot, eval = FALSE--------------------------------------------------
#  subCommands(
#  	help_head = "This is the head of the help message.",
#  
#  	"sub1", "sub1.R",
#  	        "This is the description of sub command 1, which will be a long long long text.",
#  	"sub2", "sub2.R",
#  	        "This is the description of sub command 2, which will be a long long long text.",
#  
#  	help_foot = "This is the foot of the help message."
#  )

## ----echo = FALSE, eval = TRUE, results = "asis"------------------------------
chunks <- knitr:::knit_code$get()
writeLines(c("library(GetoptLong)", chunks[["head-foot"]]), con = "main.R")

cat("```\n")
cat(system("Rscript main.R", intern = TRUE), sep = "\n")
cat("```\n")

## ----template, eval = FALSE---------------------------------------------------
#  subCommands(
#  "This is the head of the help message.
#  
#  Usage: Rscript main.R [command] [options]
#  
#  Commands:
#    <sub1=sub1.R> This is the description of sub command 1, which will be a long long
#            long text.
#    <longlonglong=sub2.R> This is the description of sub command 2, which will be a long long
#            long text.
#  
#  This is the foot of the help message.
#  "
#  )

## ----echo = FALSE, eval = TRUE, results = "asis"------------------------------
chunks <- knitr:::knit_code$get()
writeLines(c("library(GetoptLong)", chunks[["template"]]), con = "main.R")

cat("```\n")
cat(system("Rscript main.R", intern = TRUE), sep = "\n")
cat("```\n")

## ----echo = FALSE, results = "hide"-------------------------------------------
file.remove("main.R")
file.remove("sub1.R")
file.remove("sub2.R")

