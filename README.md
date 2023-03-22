# Customer-Churn-Dataset-Analysis
This project was created as part of Machine Learning Course at BZU.
After completing the analysis of the Customer Churn Data provided, the results can be found in the document provided. Based on the analysis conclusions were found that will better decision making concerning the reducing of customer churn and improving the customer’s value.

## <p>Data corelation using heat map:</p>

![image](https://user-images.githubusercontent.com/65151701/218584896-412ea09d-881e-40df-94e3-95a0557ffbb9.png)

## <p> Linear reggression models to predict customer value:</p>

![image](https://user-images.githubusercontent.com/65151701/218585013-fab797a1-11a1-4b55-a564-2625122acde2.png)

## Trained models:
<p> KNN classifier: </p>
high variance overfitting

![image](https://user-images.githubusercontent.com/65151701/218585100-0c3e9a1f-a5d5-467c-a361-003f33c390c5.png)

<p>Naive bayes classifier: </p>

![image](https://user-images.githubusercontent.com/65151701/218585183-2561c70f-8770-412d-8ecd-fdb804d13879.png)

<p>Logestic regression: </p>
high bias underfitting

![image](https://user-images.githubusercontent.com/65151701/218585266-795f7871-f2fe-47bd-8317-d7defa77913d.png)

# Conclusions:
<p>The analysis of the customer churn dataset has shown that ID and Age group attributes will not contribute to the prediction of Customer value and the classification of Churn attribute so they were dropped. </p>
<p>It is also shown, that attributes selection, in which to decide the important factors to predict “customer value” will be more accurate if chosen based on the correlation matrix, and not on the point of view of the analyzer. Which were these attributes:</p>
1-	Frequency of SMS
2-	Frequency of use
3-	Seconds of use
<p>As for the classification models, the Naive Bayes classifier was found to be the best performer among the three classification algorithms tested, because it is not an overfitting neither an under fitting model.</p>
<p>The results of this analysis can be used by your company to reduce customer churn. The consideration may be that the company starts using the Naive Bayes classifier as a tool to predict customer churn, hence prevent it. Also focus on improving the above attributes to make higher customer values. </p>
