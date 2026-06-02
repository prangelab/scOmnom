test_AtomicList_Summary <- function() {
    vec1 <- c(1L,2L,3L,5L,2L,8L)
    vec2 <- c(15L,45L,20L,1L,15L,100L,80L,5L)
    for (compress in c(TRUE, FALSE)) {
        for (type in c("IntegerList", "RleList")) {
            list1 <- do.call(type, list(one = vec1, vec2, compress = compress))
            checkIdentical(min(list1), sapply(list1, min))
            checkIdentical(max(list1), sapply(list1, max))
            checkIdentical(sum(list1), sapply(list1, sum))
            checkIdentical(mean(list1), sapply(list1, mean))
        }
    }
}

test_AtomicList_other_summarization <- function() {
    vec1 <- c(1L,2L,NA,3L,NA,5L,2L,8L)
    vec2 <- c(NA,15L,45L,20L,NA,1L,15L,100L,80L,5L,NA)
    for (compress in c(TRUE, FALSE)) {
        for (type in c("IntegerList", "RleList")) {
            list1 <- do.call(type, list(one = vec1, vec2, compress = compress))
            checkIdentical(mean(list1, na.rm=TRUE),
                           sapply(list1, mean, na.rm=TRUE))
            checkIdentical(var(list1, na.rm=TRUE),
                           sapply(list1, var, na.rm=TRUE))
            checkIdentical(sd(list1, na.rm=TRUE),
                           sapply(list1, sd, na.rm=TRUE))
            checkIdentical(median(list1, na.rm=TRUE),
                           sapply(list1, median, na.rm=TRUE))
            checkIdentical(quantile(list1, na.rm=TRUE),
                           do.call(rbind, lapply(list1, quantile, na.rm=TRUE)))
            checkIdentical(mad(list1, na.rm=TRUE),
                           sapply(list1, mad, na.rm=TRUE))
            checkIdentical(IQR(list1, na.rm=TRUE),
                           sapply(list1, IQR, na.rm=TRUE))
        }
    }
}

