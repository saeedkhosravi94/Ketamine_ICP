library(ggplot2)


plot_pca_elbow <- function(data, max_components = 50) {
  data_scaled <- scale(data)
  
  var_explained <- numeric(max_components)
  pca <- prcomp(data_scaled, center = TRUE, scale. = TRUE)
  total_variance <- sum(pca$sdev^2)
  
  for (k in 1:max_components) {
    var_explained[k] <- sum(pca$sdev[1:k]^2) / total_variance
  }
  
  df <- data.frame(
    Components = 1:max_components,
    VarianceExplained = 1 - var_explained
  )
  
  ggplot(df, aes(x = Components, y = VarianceExplained)) +
    geom_line(color = "skyblue") +
    geom_point(color = "orange") +
    xlab("Number of Components") +
    ylab("1 - Cumulative Variance Explained") +
    ggtitle("Elbow Method for PCA on Signals")
}