library(dplyr)
library(textshape)
library(stats)


# Cargar datos
dataPath <- "D:/Alexander Vivas/Documents/Universidad Ibero/Semestres/08_Semestre/inteligencia_artificial/act1_conceptos_aplicaciones_inteligencia_artificial/healthcare_dataset_stroke_data.csv"
df <- read.csv(dataPath, sep = ";")


# Limpieza de los datos
df <- textshape::column_to_rownames(df, loc = 1)
df <- df[complete.cases(df), ] 

sapply(df, class)

df <- subset(df, gender != "Other")
df$gender <- as.factor(df$gender)

nlevels(df$gender)
levels(df$gender)

df$bmi <- as.numeric(df$bmi)

df$smoking_status <- as.factor(df$smoking_status)
nlevels(df$smoking_status)
levels(df$smoking_status)

sapply(df, class)

df <- df[complete.cases(df), ]


# Reducir los datos
onlyInfarct <- df[df$stroke == 1, ]
noInfarct <- df[df$stroke == 0, ]
mediumInfarct <- noInfarct[sample(nrow(noInfarct), 209), ]
df <- data.frame(rbind(mediumInfarct, onlyInfarct))


# Entrenamiento y testeo
bound <- floor(nrow(df)/3)
train <- df[sample(nrow(df)), ]
testSet <- train[1:bound, ]
trainSet <- train[(bound + 1):nrow(train), ]


# Regresión Lineal
modelRL <- glm(stroke ~ ., data = trainSet, family = "binomial")
print(modelRL)
summary(modelRL)

# Comprobación de modelos
infarctL <- predict(modelRL, testSet)
infarctL[infarctL < 0] <- 0
infarctL[infarctL > 0] <- 1
infarctL

current_pred <- data.frame(cbind(real = testSet$stroke, regres = infarctL))
correlation_accurancy <- cor(current_pred)
correlation_accurancy

attach(current_pred)
head(current_pred)

table1 <- table(real, regres)
prop.table(table1)%>%
{. * 100} %>%
  round(2)
