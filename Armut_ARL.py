
###########################
# Business Problem
###########################

# Armut, Turkey's largest online service platform, brings together service providers and those who want to receive service.
# You can easily access services such as cleaning, renovation, transportation with a few touches on your computer or smartphone.
# provides access.
# By using the data set containing the service users and the services and categories these users have received.
# It is desired to create a product recommendation system with Association Rule Learning.

###########################
# Data set
###########################
#The dataset consists of the services customers receive and the categories of these services.
# It contains the date and time information of each service received.

# UserId: Customer number
# ServiceId: Anonymized services belonging to each category. (Example: Upholstery washing service under the cleaning category)
# A ServiceId can be found under different categories and refers to different services under different categories.
# (Example: Service with CategoryId 7 and ServiceId 4 is honeycomb cleaning, while service with CategoryId 2 and ServiceId 4 is furniture assembly)
# CategoryId: Anonymized categories. (Example: Cleaning, transportation, renovation category)
# CreateDate: The date the service was purchased

import pandas as pd
pd.set_option('display.max_columns', None)
from mlxtend.frequent_patterns import apriori, association_rules

###########################
# TASK 1: Preparing the Data
###########################

df_ = pd.read_csv("Modül_4_Tavsiye_Sistemleri/datasets/armut_data.csv")
df = df_.copy()
df.head()

# Step 2: ServiceID represents a different service for each CategoryID.
df["Hizmet"] = [str(row[1]) + "_" + str(row[2]) for row in df.values]
df.head()


# Step 3: The data set consists of the date and time the services are received, there is no basket definition (invoice, etc.).
# In order to apply Association Rule Learning, a basket (invoice, etc.) definition must be created.
# Here, the definition of basket is the services that each customer receives monthly. For example; A basket of 9_4, 46_4 services received by the customer with id 7256 in the 8th month of 2017;
# 9_4, 38_4 services received in the 10th month of 2017 represent another basket. Baskets must be identified with a unique ID.
# For this, we first create a new date variable that contains only the year and month. Set the UserID and the date variable you just created to "_"
#We combine it with and assign it to a new variable called ID.

df["CreateDate"] = pd.to_datetime(df["CreateDate"])
df.head()
df["NEW_DATE"] = df["CreateDate"].dt.strftime("%Y-%m")
df.head()
df["SepetID"] = [str(row[0]) + "_" + str(row[5]) for row in df.values]
df.head()

df[df["UserId"] == 7256 ]


###########################
# TASK 2: Generating Association Rules
###########################

# Step 1: We create the cart service pivot table as below.

# Hizmet         0_8  10_9  11_11  12_7  13_11  14_7  15_1  16_8  17_5  18_4..
# SepetID
# 0_2017-08        0     0      0     0      0     0     0     0     0     0..
# 0_2017-09        0     0      0     0      0     0     0     0     0     0..
# 0_2018-01        0     0      0     0      0     0     0     0     0     0..
# 0_2018-04        0     0      0     0      0     1     0     0     0     0..
# 10000_2017-08    0     0      0     0      0     0     0     0     0     0..

invoice_product_df = df.groupby(['SepetID', 'Hizmet'])['Hizmet'].count().unstack().fillna(0).applymap(lambda x: 1 if x > 0 else 0)
invoice_product_df.head()


# Step 2: Creating association rules.
frequent_itemsets = apriori(invoice_product_df, min_support=0.01, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="support", min_threshold=0.01)
rules.head()


#Step 3: Using the arl_recommender function to recommend a service to a user who had the last 2_0 service

def arl_recommender(rules_df, product_id, rec_count=1):
    sorted_rules = rules_df.sort_values("lift", ascending=False)
    # sorts the rules by fiber from largest to smallest. (to catch the most compatible first product)
    # sortable by confidence also depends on initiative.
    recommendation_list = [] # we create an empty list for recommended products.
    # antecedents: X
    # i: index
    # product: X, that is, the service that asks for suggestions
    for i, product in sorted_rules["antecedents"].items():
        for j in list(product):
            if j == product_id:# if the recommended product is caught:
                recommendation_list.append(list(sorted_rules.iloc[i]["consequents"]))
                # You were holding the index information with i. Add the consequents(Y) value in this index information to the recommendation_list.

    # to avoid duplication in the recommendation list:
    # For example, in 2-to-3 combinations, the same product may have dropped to the list again;
    # We take advantage of the unique feature of the dictionary structure.
    recommendation_list = list({item for item_list in recommendation_list for item in item_list})
    return recommendation_list[:rec_count] # :rec_count Get recommended items up to the desired number.



arl_recommender(rules,"2_0", 4)