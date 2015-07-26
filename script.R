# http://groupware.les.inf.puc-rio.br/har
# http://groupware.les.inf.puc-rio.br/work.jsf?p1=11201
# http://groupware.les.inf.puc-rio.br/public/papers/2013.Velloso.QAR-WLE.pdf

# https://github.com/Make42/practicalmachinelearning
# http://make42.github.io/practicalmachinelearning/assessment.html
# https://github.com/Make42/practicalmachinelearning
# https://class.coursera.org/predmachlearn-030/forum/thread?thread_id=27

library(data.table)
library(caret)
library(GGally)
library(rattle)
library(reshape2)
library(colorRamps)

setwd("~/ALPHA/Coursera/Practical Machine Learning/Assessment1/practicalmachinelearning")
training_all <- fread("pml-training.csv")
training_all[, classe:=as.factor(classe)]
training_all <- as.data.frame(training_all)
inTrain <- createDataPartition(training_all$classe, p=0.75, list=FALSE)
training <- as.data.table(training_all[inTrain,])
testing <- as.data.table(training_all[-inTrain,])
testing_final <- fread("pml-testing.csv")



############# Get rid of NA and irrelevant Features

relFeats <- names(training_all)[ which( !is.na(as.data.frame(training_all)[1,]) & as.data.frame(training_all)[1,] != "" )  ]
relFeats <- relFeats[ 8:length(relFeats) ]
relFeats <- relFeats[ !(relFeats %in% c("classe")) ]


############# FEATURE SELECTION

##### Near Zero Variance

## Calc

freqCut <- 95/5
uniqueCut <- 10

nzv <- nearZeroVar(training[,relFeats,with=F], freqCut, uniqueCut, saveMetrics = TRUE)
nzv$featurename <- row.names(nzv)
nzv <- as.data.table(nzv)

## Plot

gg_nzv <- ggplot(nzv, aes(x=freqRatio, y=percentUnique)) + # , color=freqRatio, fill=percentUnique
  geom_hline(yintercept=uniqueCut) +
  geom_vline(xintercept=freqCut) +
  geom_point(size=2) +
#   scale_colour_continuous(low = "green", high="red", trans = "log") + # freqRatio - klein ist gut
#   scale_fill_continuous(low = "red", high="green", trans = "log") + # percentUnique - groß ist gut
  scale_shape_manual(values=c(21,22)) +
  scale_x_log10() +
  scale_y_log10() +
  geom_text(aes(label=ifelse(freqRatio<freqCut & percentUnique>uniqueCut, featurename, '')), 
            hjust=0, vjust=0, angle = -90, color="black", size=4)

## Include in feature Analysis

featAn <- nzv

##### Correlation

## Calc

corMatr <- cor(training[,relFeats,with=F])
corVals <- sort(abs(unique(c(corMatr))))
corVals <- corVals[1:length(corVals)-1]

## Plot

gg_cor <- ggplot(data.frame(corVals = corVals, int = 1:length(corVals)), aes(x=corVals)) + geom_histogram()

## Include in feature Analysis

featAn$corMetr <- 0.05
for ( cut in seq(0.1,0.9,0.1) ) {
  highCor <- findCorrelation(corMatr, cutoff=cut, names=T)
  featAn$corMetr[ featAn$featurename %in% highCor ] <- cut+0.05
}

#### Select Feats

## Print Features

gg_feats <- ggplot(featAn, aes(x=corMetr, y=percentUnique, color=freqRatio)) +
  geom_hline(yintercept=1) +
  geom_vline(xintercept=0.6) +
  geom_point(size=4) +
  scale_colour_continuous(low = "green", high="red", trans = "log") + # freqRatio - klein ist gut
  scale_shape_manual(values=c(21,22)) +
  scale_y_log10()

## Select

relFeats2 <- featAn[freqRatio<7 &
                    percentUnique>1 &
                    !( featurename %in% findCorrelation(corMatr, cutoff=0.6, names=T) ),
                    featurename]

## Print

for (colname in relFeats2 ) {
  
  data_feature <- training[,.(get(colname),classe)]
  setnames(data_feature, c("V1"),colname)
  
  gg <- ggplot(data=data_feature, mapping=aes(x=get(colname), fill=classe, color = classe)) + 
    # geom_histogram(alpha = 0.3, position = "identity") +
    geom_density(alpha=1/3) +
    labs(x=colname)  # +
  # scale_y_log10()
  #     scale_colour_manual(values=genuine_fraud_cols) +
  #     scale_fill_manual(values=genuine_fraud_cols)
  #   if ( is.factor(data_feature[, get(colname)]) & length(levels(data_feature[, get(colname)])) > 10 ) {
  #     gg <- gg + theme( axis.text.x = element_blank(),
  #                       axis.ticks = element_blank() ) # axis.text.x = element_text(angle = -90, hjust = 0)
  #   } else {
  #     gg <- gg + scale_x_log10()
  #   }
  
  # print(gg)
}

# gg_pairs <- ggpairs( data = training,
#                columns = which(names(training) %in% relFeats2),
#                color = "classe",
#                params = c("shape" = 1),
#                upper = list(continuous = "density") )
# for (i in 1:length(gg$plots)) {
#   gg$plots[[i]] <- paste(gg$plots[[i]],
#                          "+ scale_x_log10()",
#                          "+ scale_y_log10()" )
# }


############# TRAINING & CROSS-VALIDATION

fitModel <- train(classe ~ ., training[, c("classe", relFeats2), with = FALSE], method = "rpart")


confMatr_train <- confusionMatrix(predict(fitModel, newdata = training), training$classe)
confMatr_train

confMatr_test <- confusionMatrix(predict(fitModel, newdata = testing), testing$classe)
confMatr_test

confMatr_train_dt <- data.table(confMatr_train$table)
confMatr_train_gf <- data.table(
  Prediction = c("A", "A", "nA", "nA"),
  Reference = c("A", "nA", "A", "nA"),
  value = c(
    confMatr_train_dt[Reference == "A" & Prediction == "A", sum(N)],
    confMatr_train_dt[Reference == "A" & Prediction != "A", sum(N)],
    confMatr_train_dt[Reference != "A" & Prediction == "A", sum(N)],
    confMatr_train_dt[Reference != "A" & Prediction != "A", sum(N)]
  )
)

confMatr_test_dt <- data.table(confMatr_test$table)
confMatr_test_gf <- data.table(
  Prediction = c("A", "A", "nA", "nA"),
  Reference = c("A", "nA", "A", "nA"),
  value = c(
    confMatr_test_dt[Reference == "A" & Prediction == "A", sum(N)],
    confMatr_test_dt[Reference == "A" & Prediction != "A", sum(N)],
    confMatr_test_dt[Reference != "A" & Prediction == "A", sum(N)],
    confMatr_test_dt[Reference != "A" & Prediction != "A", sum(N)]
  )
)

gg_conf1 <- ggplot(melt(confMatr_train$table), aes(x=Reference,y=Prediction,color=value)) +
  geom_text(aes(label=as.character(round(value, digits = 2))), size=6) +
  scale_color_gradientn(colours=matlab.like2(9))

gg_conf2 <- ggplot(melt(confMatr_test$table), aes(x=Reference,y=Prediction,color=value)) +
  geom_text(aes(label=as.character(round(value, digits = 2))), size=6) +
  scale_color_gradientn(colours=matlab.like2(9))

gg_conf1_gf <- ggplot(confMatr_train_gf, aes(x=Reference,y=Prediction,color=value)) +
  geom_text(aes(label=as.character(round(value, digits = 2))), size=6) +
  scale_color_gradientn(colours=matlab.like2(9))

gg_conf2_gf <- ggplot(confMatr_test_gf, aes(x=Reference,y=Prediction,color=value)) +
  geom_text(aes(label=as.character(round(value, digits = 2))), size=6) +
  scale_color_gradientn(colours=matlab.like2(9))


############# TESTING

finalpredict <- predict(fitModel, newdata = testing_final)



############# FEATURES

"avg_roll_belt"
"var_roll_belt"
"max accelometer belt"
"range accelometer belt"
"var_accel_belt"
"var gyro belt"
"var magnetometer belt"
"variance accelometer arm"
"max magnetometer arm"
"min magnetometer arm"
"max acceleration dumbell"
"var gyro dumbell"
"min acceleration  dumbell"
"sum of the pitch glove"
"max gyro glove"
"min gyro glove"

# ggplot(training, aes(x=avg_roll_belt, y=var_roll_belt, color=classe)) + geom_point()
