/* Import data from a CSV file */
proc import datafile='/path/to/your/data.csv' 
    out=mydata
    dbms=csv 
    replace;
    getnames=yes;
run;

/* Split the dataset into training (70%) and testing (30%) subsets */
proc surveyselect data=mydata out=train_test split=70 seed=42;
    sampsize 70; /* Specify the sample size for training */
run;

/* Create training and testing datasets */
data train test;
    set train_test;
    if selected = 1 then output train; /* Training data */
    else output test; /* Testing data */
run;

/* Generate frequency distribution for the ACCOUNT_TYPE variable */
proc freq data=train;
    tables ACCOUNT_TYPE / out=freq_account_type;
run;

/* Calculate Information Value (IV) for each bin */
data iv_account_type;
    set freq_account_type;
    /* Insert calculations for IV and other relevant metrics here */
run;

/* Perform Logistic Regression with Stepwise Selection */
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
    output out=predicted p=prob; /* Output predicted probabilities */
run;

/* Evaluate the performance of the model */
proc freq data=predicted;
    tables LI_FLAG*prob / chisq; /* Chi-square test for LI_FLAG vs. predicted probabilities */
run;

/* Manually compute sensitivity and recall */
data metrics;
    set predicted;
    /* Convert probabilities to binary outcomes */
    predicted_class = (prob >= 0.5);
    
    /* Initialize confusion matrix components */
    retain TP 0 FN 0 FP 0 TN 0;

    /* Update confusion matrix counts */
    if LI_FLAG = 1 and predicted_class = 1 then TP + 1; /* True Positive */
    else if LI_FLAG = 1 and predicted_class = 0 then FN + 1; /* False Negative */
    else if LI_FLAG = 0 and predicted_class = 1 then FP + 1; /* False Positive */
    else if LI_FLAG = 0 and predicted_class = 0 then TN + 1; /* True Negative */

    /* Calculate sensitivity and recall */
    sensitivity = TP / (TP + FN);
    recall = TP / (TP + FN);
    
    /* Display metrics */
    put "Sensitivity: " sensitivity;
    put "Recall: " recall;
run;

/* Score the test dataset using the logistic regression model */
proc logistic inmodel=train descending;
    score data=test out=scored_test; /* Output scored test dataset */
run;

/* Create decile ranks based on predicted probabilities */
proc rank data=scored_test out=ranked_test groups=10;
    var prob; /* Use the probability variable from scoring output */
    ranks decile;
run;

/* Calculate the average LI_FLAG for each decile */
proc means data=ranked_test noprint;
    class decile;
    var LI_FLAG;
    output out=decile_stats mean=mean_LI_FLAG; /* Output mean LI_FLAG by decile */
run;

/* Plot the Decile Chart */
proc sgplot data=decile_stats;
    series x=decile y=mean_LI_FLAG / markers; /* Create a series plot */
    xaxis label="Decile";
    yaxis label="Mean LI_FLAG";
run;
