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

theme_linedraw(
  base_size = 11,
  base_family = "",
  base_line_size = base_size / 22,
  base_rect_size = base_size / 22
)

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
V4 <- c(125, 130, 135,140)
V5 <- c(165, 170, 175, 170)
V6 <- c(205, 210, 215, 220)
V7 <- c(245, 250, 255, 260)


plot_series <- function(Data, Class, V)
{
  time <- 1:60
  plot_list <- list()
  for (i in 1:length(V))
    {
    plot_list[[i]] <- local(
      {
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
        plot_series(TS1, TS1_Class, V7))

Data1 <- wrap_plots(p1, ncol = 4) + plot_layout(guides = "collect")

#Dataset2 Visualization

TimeSeries2 <- read.csv(file = "TimeSeries2.csv",
                        header = TRUE, row.names = 1)

time <- 1:2800

plot1 <- ggplot(data = TimeSeries2, mapping = aes(x = time, y = F1))
plot1 <- plot1 + geom_line(alpha = 0.3) +
  geom_point(aes(x = time, y = LabelF1, color = as.factor(LabelF1))) +
  scale_color_discrete(name = "Classification", labels = c("IC", "OC")) +
  theme_classic()

plot2 <- ggplot(data = TimeSeries2, mapping = aes(x = time, y = F2))
plot2 <- plot2 + geom_line(alpha = 0.3) +
  geom_point(aes(x = time, y = LabelF2, color = as.factor(LabelF2))) +
  scale_color_discrete(name = "Classification", labels = c("IC", "OC")) +
  theme_classic()

plot3 <- ggplot(data = TimeSeries2, mapping = aes(x = time, y = F3))
plot3 <- plot3 + geom_line(alpha = 0.3) +
  geom_point(aes(x = time, y = LabelF3, color = as.factor(LabelF3))) +
  scale_color_discrete(name = "Classification", labels = c("IC", "OC")) +
  theme_classic()

plot4 <- ggplot(data = TimeSeries2, mapping = aes(x = time, y = F4))
plot4 <- plot4 + geom_line(alpha = 0.3) +
  geom_point(aes(x = time, y = LabelF4, color = as.factor(LabelF4))) +
  scale_color_discrete(name = "Classification", labels = c("IC", "OC")) +
  theme_classic()

plot5 <- ggplot(data = TimeSeries2, mapping = aes(x = time, y = F5))
plot5 <- plot5 + geom_line(alpha = 0.3) +
  geom_point(aes(x = time, y = LabelF5, color = as.factor(LabelF5))) +
  scale_color_discrete(name = "Classification", labels = c("IC", "OC")) +
  theme_classic()
  
layout <- "
A#B#C
#D#E#
"
Data2 <- wrap_plots(plot1, plot2, plot3, plot4, plot5, ncol = 3, nrow = 2,
                    design = layout) +
  plot_layout(guides = "collect")