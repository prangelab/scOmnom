###

test_find_core_IRanges_cols <- function()
{
    find_core_IRanges_cols <- IRanges:::find_core_IRanges_cols

    df_colnames <- c("stop", "chrom", "start")
    target <- c(start=3L, end=1L, width=NA_integer_)
    current <- find_core_IRanges_cols(df_colnames)
    checkIdentical(target, current)

    df_colnames <- c("stop", "width", "start", "AA")
    target <- c(start=3L, end=1L, width=2L)
    current <- find_core_IRanges_cols(df_colnames)
    checkIdentical(target, current)

    df_colnames <- c("AA", "START", "END")
    target <- c(start=2L, end=3L, width=NA_integer_)
    current <- find_core_IRanges_cols(df_colnames)
    checkIdentical(target, current)

    df_colnames <- c("..", "txEnd", "..", "txStart")
    target <- c(start=4L, end=2L, width=NA_integer_)
    current <- find_core_IRanges_cols(df_colnames)
    checkIdentical(target, current)

    df_colnames <- c("chrom", "strand", "exon_chrom_start", "exon_chrom_end")
    target <- c(start=3L, end=4L, width=NA_integer_)
    current <- find_core_IRanges_cols(df_colnames)
    checkIdentical(target, current)

    df_colnames <- c("AA", "start", "end", "stop")
    checkException(find_core_IRanges_cols(df_colnames), silent=TRUE)
    target <- c(start=2L, end=4L, width=NA_integer_)
    current <- find_core_IRanges_cols(df_colnames, end.field="stop")
    checkIdentical(target, current)
    checkException(find_core_IRanges_cols(df_colnames, end.field=4),
                   silent=TRUE)
}

test_makeIRangesFromDataFrame <- function() {
    df <- data.frame(aa=11:15, end=c(31, NA, 33:35), start=21:25)
    rownames(df) <- LETTERS[1:5]
    checkException(makeIRangesFromDataFrame(df), silent=TRUE)
    current <- makeIRangesFromDataFrame(df, keep.extra.columns=TRUE, na.rm=TRUE)
    target <- IRanges(21:25, 31:35, names=LETTERS[1:5], aa=11:15)[-2]
    checkIdentical(current, target)
}

