if (!require("ggplot2")) {
  install.packages("ggplot2")
  library("ggplot2")
}

if (!require("egg")) {
  install.packages("egg")
  library("egg")
}

if (!require("patchwork")) {
  install.packages("patchwork")
  library("patchwork")
}

#Dataset1 Visualization
TimeSeries1 <- read.csv(file = "TimeSeries1.csv",
                        header = TRUE, row.names = 1)
TimeSeries1_Classification <- read.csv(file = "TimeSeries1_Classification.csv",
                        header = TRUE, row.names = 1)

TS1 <- as.data.frame(t(TimeSeries1))
TS1_Class <- as.data.frame(t(TimeSeries1_Classification))

V1 <- c(5, 10, 15, 20)
V2 <- c(45, 50, 55, 60)
V3 <- c(85, 90, 95, 100)
V4 <- c(125, 130, 135, 140)
V5 <- c(165, 170, 175, 170)
V6 <- c(205, 210, 215, 220)
V7 <- c(245, 250, 255, 260)
V8 <- c(285, 290, 295, 300)


plot_series <- function(Data, Class, V)
{
  time <- 1:60
  plot_list <- list()
  for (i in 1:length(V)){
    plot_list[[i]] <- local({
      i <- i
      p <- ggplot(data = Data, mapping = aes(x = time, y = Data[, V[i]]))
      p <- p + geom_line(alpha = 0.3) +
        geom_point(data = Class, aes(x = time, y = Data[, V[i]],
                                    color = as.factor(Class[, V[i]])),
                                    alpha = 0.8, shape = 21) +
        scale_color_discrete(name = "Classification", labels = c("IC", "OC")) +
        theme_classic()
      })
    }
  return(plot_list)
}

p1 <- c(plot_series(TS1, TS1_Class, V1),
        plot_series(TS1, TS1_Class, V2),
        plot_series(TS1, TS1_Class, V3),
        plot_series(TS1, TS1_Class, V4),
        plot_series(TS1, TS1_Class, V5),
        plot_series(TS1, TS1_Class, V6),
        plot_series(TS1, TS1_Class, V7),
        plot_series(TS1, TS1_Class, V8))

Data1 <- wrap_plots(p1, ncol = 4) + plot_layout(guides = "collect")

#Dataset1 with WE anomalies visualization

WE_TS1_Class <- read.csv(file = "TimeSeries1_WE_Classification.csv",
                        header = TRUE, row.names = 1)

p12 <- c(plot_series(TS1, WE_TS1_Class, V1),
        plot_series(TS1, WE_TS1_Class, V2),
        plot_series(TS1, WE_TS1_Class, V3),
        plot_series(TS1, WE_TS1_Class, V4),
        plot_series(TS1, WE_TS1_Class, V5),
        plot_series(TS1, WE_TS1_Class, V6),
        plot_series(TS1, WE_TS1_Class, V7),
        plot_series(TS1, WE_TS1_Class, V8))

Data12 <- wrap_plots(p12, ncol = 4) + plot_layout(guides = "collect")
