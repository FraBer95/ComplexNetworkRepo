library(survival)
library(survminer)
library(survex)
library(randomForestSRC)
library(ggsurvfit)
library(ggplot2)
library(pec)
library(caret)
library(survivalsvm)
library(SurvMetrics)
library(gbm)
library(mlr3proba)
library(mlr3extralearners)
library(mlr3pipelines)
library(paradox)
library(mlr3tuning)
library(survivalmodels)


print('Starting...')
data_train <- read.csv("dataset//data_CPAP_train.csv")
data_test <- read.csv("dataset//data_CPAP_test.csv")

data_train <- lapply(data_train, as.numeric)
data_train <- as.data.frame(data_train)
data_test <- lapply(data_test, as.numeric)
data_test <- as.data.frame(data_test)

data_train$Durata.follow.up.da.dimissione<-data_train$Durata.follow.up.da.dimissione/30
data_test$Durata.follow.up.da.dimissione<-data_test$Durata.follow.up.da.dimissione/30

data_train <- data_train[, -c(4,24,26,27,28,29)]
data_test <- data_test[, -c(4,24,26, 27,28,29)]






cox_train <- TRUE
expl_cox <- TRUE

rf_train <- FALSE
expl_rf <- FALSE

time <- data_train$Durata.follow.up.da.dimissione
status <- data_train$Status


find_categorical_variables <- function(data) {
  categorical_vars <- character()  # Inizializza un vettore vuoto per le variabili categoriche

  for (col in names(data)) {
    if (is.factor(data[[col]]) || is.character(data[[col]])) {
      categorical_vars <- c(categorical_vars, col)
    } else if (is.numeric(data[[col]])) {
      unique_values <- unique(data[[col]])
      if (length(unique_values) < 10) {  # Arbitrariamente, consideriamo le variabili con meno di 10 valori unici come categoriche
        categorical_vars <- c(categorical_vars, col)
      }
    }
  }

  return(categorical_vars)
}

plot_sc <- FALSE
if (plot_sc){
  not_col <- c('AHI', 'SaO2.min', 'Years_of_CPAP', 'CPAP_0_5','CPAP_5_10', 'CPAP.10')
  columns <- names(data)
  temp <- data[,!columns %in% not_col]
  columns <- names(temp)
  time <- data$Durata.follow.up.da.dimissione
  status <- data$Status
  print("Plotting Survival Curves...")
  for (feature in columns[3:length(columns)]){
  fit  <- survfit(Surv(time,status) ~ data[[feature]])
  n_val <- length(unique(data[[feature]]))
  values <- unique(data[[feature]])
  labs <- character(0)
  for(i in 1:n_val){
    string <- paste(feature," = ", i-1)
    labs[i] <- string
  }
  print(feature)
  p <- ggsurvplot(
    fit,
    data = data,
    risk.table = TRUE,
    pval = TRUE,
    main = paste("Curva di sopravvivenza per", feature),
    conf.int = TRUE,
    legend.labs = labs,
    legend.title = feature)

  print(p)
}
}

if (cox_train){
  print("Training COX Model...")
  cox_model <- coxph(Surv(time,status) ~ ., data =  data_train[, -c(1,2)], x=TRUE, model = TRUE)
  cox_summary <- summary(cox_model)
  cox_explainer <- survex::explain(cox_model,data =  data_train[, -c(1,2)],
                                   y = survival::Surv(data_train$Durata.follow.up.da.dimissione, data_train$Status),
                                   verbose=FALSE)
  cox_explainer_test <- survex::explain(cox_model,data = data_test[,-c(1,2)],
                                 y = survival::Surv(data_test$Durata.follow.up.da.dimissione, data_test$Status),
                                 verbose=FALSE)


  if (expl_cox){
    print("---> Explenability COX Model...")
    modelparts_cox <- model_parts(cox_explainer_test, output_type="survival")
    plot(modelparts_cox)


  print("Cox Terminated!")
}

if(rf_train){
  print("Training SURVIVAL RANDOM FOREST Model ...")
  formula <- Surv(Durata.follow.up.da.dimissione,Status) ~ .
  model_rf <- rfsrc(formula, data = data_train[, -c(4,24,26,27,28,29)], ntree = 50, nsplit = 10)
  rf_explainer <- survex::explain(model_rf,data =  data_train[, -c(4,24,26,27,28,29)],
                                   y = survival::Surv(data_train$Durata.follow.up.da.dimissione, data_train$Status),
                                   verbose=FALSE)
  rf_explainer_test <- survex::explain(model_rf,data =  data_test[,-c(4,24,26,27,28,29)],
                                 y = survival::Surv(data_test$Durata.follow.up.da.dimissione, data_test$Status),
                                 verbose=FALSE)
  if(expl_rf){
    print("---> Explenability SRF Model...")
    modelparts_rf <- model_parts(rf_explainer_test,type = "variable_importance", output_type="survival")
    save.image("Rdata\\modelparts_RF.Rdata")

  }
  print("SRF Terminated!")
}


# Evaluation Metrics
print("Evaluating Models... ")

cox_perf <- model_performance(cox_explainer_test)
rf_perf <- model_performance(rf_explainer_test, new_observation = data_test[, -c(1,2,4,24,26,27,28,29)])


plot(cox_perf,rf_perf,metrics_type = "scalar")
#save.image("Rdata\\workspace_analysis.Rdata")