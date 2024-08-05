/* Import data from CSV */
proc import datafile='/path/to/your/data.csv' 
    out=mydata
    dbms=csv 
    replace;
    getnames=yes;
run;

/* Split data into training (70%) and test (30%) */
proc surveyselect data=mydata out=train_test split=70 seed=42;
    sampsize 70;
run;

data train;
    set train_test;
    if selected = 1;
run;

data test;
    set train_test;
    if selected = 0;
run;

/* Create bins for categorical variables (example for ACCOUNT_TYPE) */
proc freq data=train;
    tables ACCOUNT_TYPE / out=freq_account_type;
run;

/* Calculate Information Value for each bin */
data iv_account_type;
    set freq_account_type;
    /* Calculate IV and other necessary metrics here */
run;

/* Logistic Regression with Stepwise Selection */
proc logistic data=train descending;
    class ACCOUNT_TYPE (param=ref) CUSTOMER_TAG (param=ref) EDUCATION_LEVEL (param=ref) 
          GENDER (param=ref) INTERNET_BANKING_USAGE (param=ref) MARITAL_STATUS (param=ref) 
          NOMINEE_AVAILABLE_FLAG (param=ref) RM_ALLOCATED_FLAG (param=ref) OCCUPATION (param=ref) 
          STATE (param=ref);
    model LI_FLAG(event='1') = AGE AQB_BALANCE AUM CIBIL_SCORE CREDIT_CARD_LIMIT
                                  CR_AMT_12MNTH CR_CNT_12MNTH DC_APPAREL_30DAYS_ACTV
                                  DC_ECOM_30DAYS_AMT DC_ECOM_30DAYS_CNT DC_FOOD_30DAYS_ACTV
                                  DC_FUEL_30DAYS_ACTV DC_GROCRY_30DAYS_ACTV DC_OTT_30DAYS_ACTV
                                  DC_POS_30DAYS_AMT DC_POS_30DAYS_CNT DC_RECHARGE_30DAYS_ACTV
                                  DC_TRAVEL_30DAYS_ACTV DC_UTILITY_30DAYS_ACTV DR_AMT_12MNTH
                                  DR_CNT_12MNTH DR_CR_RATIO FD_COUNT FD_CURRENTMONTHANR
                                  FD_FLAG GI_FLAG HEALTH_FLAG INCOME_NET KYC_LAST_DONE_DATE
                                  LI_FLAG MASS_FLAG MBK_ACTIVE MF_FLAG MONTHLY_BALANCE NRV
                                  NR_FLAG TOTAL_LIVE_SECURED_AMT TOTAL_LIVE_UNSECURED_AMT
                                  VINTAGE_DAYS / selection=stepwise slentry=0.05 slstay=0.05;
    output out=predicted p=prob;
run;

/* Evaluate model performance */
proc freq data=predicted;
    tables LI_FLAG*prob / chisq;
run;

/* Compute sensitivity and recall manually */
data metrics;
    set predicted;
    /* Convert probabilities to binary outcomes */
    predicted_class = (prob >= 0.5);
    
    /* Confusion Matrix components */
    if LI_FLAG = 1 and predicted_class = 1 then TP + 1;
    else if LI_FLAG = 1 and predicted_class = 0 then FN + 1;
    else if LI_FLAG = 0 and predicted_class = 1 then FP + 1;
    else if LI_FLAG = 0 and predicted_class = 0 then TN + 1;

    /* Calculate sensitivity and recall */
    sensitivity = TP / (TP + FN);
    recall = TP / (TP + FN);
    
    /* Display metrics */
    put "Sensitivity: " sensitivity;
    put "Recall: " recall;
run;

/* Score test data using the logistic regression model */
proc logistic inmodel=train descending;
    score data=test out=scored_test;
run;

/* Create Decile Plot */
proc rank data=scored_test out=ranked_test groups=10;
    var prob; /* or the probability variable from the scoring output */
    ranks decile;
run;

/* Calculate the average LI_FLAG by decile */
proc means data=ranked_test noprint;
    class decile;
    var LI_FLAG;
    output out=decile_stats mean=mean_LI_FLAG;
run;

/* Plot the Decile Chart */
proc sgplot data=decile_stats;
    series x=decile y=mean_LI_FLAG / markers;
    xaxis label="Decile";
    yaxis label="Mean LI_FLAG";
run;



