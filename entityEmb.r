library(dplyr)
library(caret)
library(darch)

# Benchmark entity embedding

# Entrainer une random forest sur le dataset "adult"

adult <- read.csv("~/Projets/TestSpark/adult.csv", na.strings = "?", stringsAsFactors = T)
adult <- adult[complete.cases(adult),]

adult <- adult %>% select(-education_num)

adult <- adult[sample(1:nrow(adult), nrow(adult)), ]

y <- adult$income
adult$income <- NULL

high_cardinality_vars <- c(4,6,13) # Education / Occupation / Country i.e. bcp de modalités

# Train / Test split
set.seed(1234)
train_idx <- createDataPartition(y, p = 0.8)
train_dat <- adult[train_idx$Resample1, ]
test_dat <- adult[-train_idx$Resample1, ]

y_train <- ifelse(y[train_idx$Resample1] == ">50K", 1, 0)
y_test <- ifelse(y[-train_idx$Resample1] == ">50K", 1, 0)


# Center Scale numerical data
preProcValues <- preProcess(train_dat, method = c("center", "scale"))

train_dat <- predict(preProcValues, train_dat)
test_dat <- predict(preProcValues, test_dat)

# End of benchmark

##########################################################################################
##########################################################################################

# INPUT
# train_dat : a data.frame
# test_dat  : a data.frame
# y_train   : a vector
# high_cardinality_vars : indices of variables with high cardinality

# OUTPUT
# train_dat_embed : a data.frame
# test_dat_embed  : a data.frame


# Encodage par entity embedding des categories
nb_var <- length(high_cardinality_vars)

if(nb_var != 1){
  nb_mods_by_variable <- apply(train_dat[,high_cardinality_vars], MARGIN = 2, n_distinct)
}else
{
  nb_mods_by_variable <- n_distinct(train_dat[,high_cardinality_vars])
}



A <- 10   # 15 : more complex
B <- 5    # 2 or 3 : more complex

# number of neurons for layer 1 et 2
n_layer1 <- min(1000,as.integer((A*length(high_cardinality_vars)**0.5)*sum(log(nb_mods_by_variable))+1))    # tuning
n_layer2 <- as.integer(n_layer1/B + 2)

#dropouts
dropout1 <- 0.1
dropout2 <- 0.1

#learning parameters
epochs <- 30  #25 : more iterations
batch_size <- 256 # 256 : gradient more stable

#########################################################

# Pour la couche embedding, 
# 1 -  on regarde pour chaque variable le nombre de modalités
# 2 -  le nombre de neurons embedding est donné par : int(5*(1-exp(-nb_modalite*0.05)))+1
#      entre 2 et 5 en général


computeEmbedNeuronSize <- function(nb_modalite){
  as.integer(5*(1-exp(-nb_modalite*0.05)))+1
}

nb_embed_neuron_size <- sum(sapply(nb_mods_by_variable, FUN = computeEmbedNeuronSize))

###############################################################################

# Construction du réseau de neurones
#
# Couche d'entrée ==> couche embedding ==> lay1 ==> lay2 ==> Sortie

dummyVarMod <- train_dat %>%
  select(high_cardinality_vars)%>%
  dummyVars(~., ., fullRank = T)

x <- predict(dummyVarMod , train_dat)
x_test <- predict(dummyVarMod, test_dat)


layers <- c(ncol(x), nb_embed_neuron_size, n_layer1, n_layer2, 1)

dropout <- c(0, dropout1, dropout2,0)

darch_object <- darch(
  x, y_train, layers,
  darch.isClass = T,
  darch.dropout = dropout,
  darch.unitFunction = c("rectifiedLinearUnit", "rectifiedLinearUnit", "rectifiedLinearUnit", "sigmoidUnit"),
  darch.batchSize = batch_size,
  darch.numEpochs = epochs
)

# On retourne la sortie de la couche 2 du réseau
out <- as.data.frame(predict(darch_object, x, outputLayer = 2))
out_test <- as.data.frame(predict(darch_object, x_test, outputLayer = 2))


# On droppe les categorical vars et on les remplace par la sortie du réseau
train_dat_embed <- train_dat %>%
  select(-high_cardinality_vars)%>%
  bind_cols(out)

test_dat_embed <- test_dat %>%
  select(-high_cardinality_vars)%>%
  bind_cols(out_test)

##################################################################################
