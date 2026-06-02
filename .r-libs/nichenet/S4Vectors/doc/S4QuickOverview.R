## ----setup, echo=FALSE, results='hide'--------------------
options(width=60)
suppressPackageStartupMessages({
    library(Matrix)
    library(IRanges)
    library(ShortRead)
    library(graph)
})

## ----S4_object_in_dataset---------------------------------
library(graph)
data(apopGraph)
apopGraph

## ----S4_object_from_constructor---------------------------
library(IRanges)
IRanges(start=c(101, 25), end=c(110, 80))

## ----S4_object_from_ceorcion------------------------------
library(Matrix)
m <- matrix(3:-4, nrow=2)
as(m, "Matrix")

## ----eval=FALSE-------------------------------------------
# library(GenomicFeatures)
# makeTxDbFromUCSC("sacCer2", tablename="ensGene")

## ----S4_object_from_high_level_IO_function----------------
library(ShortRead)
path_to_my_data <- system.file(
    package="ShortRead",
    "extdata", "Data", "C1-36Firecrest", "Bustard", "GERALD")
lane1 <- readFastq(path_to_my_data, pattern="s_1_sequence.txt")
lane1

## ----S4_object_inside_another_object----------------------
sread(lane1)

## ----getters_and_setters----------------------------------
ir <- IRanges(start=c(101, 25), end=c(110, 80))
width(ir)
width(ir) <- width(ir) - 5
ir

## ----specialized_methods----------------------------------
qa1 <- qa(lane1, lane="lane1")
class(qa1)

## ----showMethods------------------------------------------
showMethods("qa")

## ----showClass, R.options=list(width=60)------------------
class(lane1)
showClass("ShortReadQ")

## ----setClass---------------------------------------------
setClass("SNPLocations",
    slots=c(
      genome="character",  # a single string
      snpid="character",   # a character vector of length N
      chrom="character",   # a character vector of length N
      pos="integer"        # an integer vector of length N
    )
)

## ----SNPLocations-----------------------------------------
SNPLocations <- function(genome, snpid, chrom, pos)
    new("SNPLocations", genome=genome, snpid=snpid, chrom=chrom, pos=pos)

## ----test_SNPLocations------------------------------------
snplocs <- SNPLocations("hg19",
             c("rs0001", "rs0002"),
             c("chr1", "chrX"),
             c(224033L, 1266886L))

## ----length, results='hide'-------------------------------
setMethod("length", "SNPLocations", function(x) length(x@snpid))

## ----test_length------------------------------------------
length(snplocs)  # just testing

## ----genome, results='hide'-------------------------------
setGeneric("genome", function(x) standardGeneric("genome"))
setMethod("genome", "SNPLocations", function(x) x@genome)

## ----snpid, results='hide'--------------------------------
setGeneric("snpid", function(x) standardGeneric("snpid"))
setMethod("snpid", "SNPLocations", function(x) x@snpid)

## ----chrom, results='hide'--------------------------------
setGeneric("chrom", function(x) standardGeneric("chrom"))
setMethod("chrom", "SNPLocations", function(x) x@chrom)

## ----pos, results='hide'----------------------------------
setGeneric("pos", function(x) standardGeneric("pos"))
setMethod("pos", "SNPLocations", function(x) x@pos)

## ----test_slot_getters------------------------------------
genome(snplocs)  # just testing
snpid(snplocs)   # just testing

## ----show, results='hide'---------------------------------
setMethod("show", "SNPLocations",
    function(object)
        cat(class(object), "instance with", length(object),
            "SNPs on genome", genome(object), "\n")
)

## ---------------------------------------------------------
snplocs  # just testing

## ----validity, results='hide'-----------------------------
setValidity("SNPLocations",
    function(object) {
        if (!is.character(genome(object)) ||
            length(genome(object)) != 1 || is.na(genome(object)))
            return("'genome' slot must be a single string")
        slot_lengths <- c(length(snpid(object)),
                          length(chrom(object)),
                          length(pos(object)))
        if (length(unique(slot_lengths)) != 1)
            return("lengths of slots 'snpid', 'chrom' and 'pos' differ")
        TRUE
    }
)

## ----error=TRUE-------------------------------------------
try({
snplocs@chrom <- LETTERS[1:3]  # a very bad idea!
validObject(snplocs)
})

## ----set_chrom, results='hide'----------------------------
setGeneric("chrom<-", function(x, value) standardGeneric("chrom<-"))
setReplaceMethod("chrom", "SNPLocations",
    function(x, value) {x@chrom <- value; validObject(x); x})

## ----test_slot_setters------------------------------------
chrom(snplocs) <- LETTERS[1:2]  # repair currently broken object

## ----error=TRUE-------------------------------------------
try({
chrom(snplocs) <- LETTERS[1:3]  # try to break it again
})

## ----setAs, results='hide'--------------------------------
setAs("SNPLocations", "data.frame",
    function(from)
        data.frame(snpid=snpid(from), chrom=chrom(from), pos=pos(from))
)

## ----test_coercion----------------------------------------
as(snplocs, "data.frame")  # testing

## ----AnnotatedSNPs----------------------------------------
setClass("AnnotatedSNPs",
    contains="SNPLocations",
    slots=c(
        geneid="character"  # a character vector of length N
    )
)

## ----slot_inheritance-------------------------------------
showClass("AnnotatedSNPs")

## ----AnnotatedSNPs_constructor----------------------------
AnnotatedSNPs <- function(genome, snpid, chrom, pos, geneid)
{
    new("AnnotatedSNPs",
        SNPLocations(genome, snpid, chrom, pos),
        geneid=geneid)
}

## ----method_inheritance-----------------------------------
snps <- AnnotatedSNPs("hg19",
             c("rs0001", "rs0002"),
             c("chr1", "chrX"),
             c(224033L, 1266886L),
             c("AAU1", "SXW-23"))

## ----method_inheritance_2---------------------------------
snps

## ----as_data_frame_is_not_right---------------------------
as(snps, "data.frame")  # the 'geneid' slot is ignored

## ---------------------------------------------------------
is(snps, "AnnotatedSNPs")     # 'snps' is an AnnotatedSNPs object
is(snps, "SNPLocations")      # and is also a SNPLocations object
class(snps)                   # but is *not* a SNPLocations *instance*

## ----automatic_coercion_method----------------------------
as(snps, "SNPLocations")

## ----incremental_validity_method, results='hide'----------
setValidity("AnnotatedSNPs",
    function(object) {
        if (length(object@geneid) != length(object))
            return("'geneid' slot must have the length of the object")
        TRUE
    }
)

