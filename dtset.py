#import pandas as pd

#df = pd.read_csv("enron_spam_data.csv")
#df = df.rename(columns={"Spam/Ham": "spam"})
#df = df.rename(columns={"Message": "text"})

#df["spam"] = df["spam"].map({"spam": 1, "ham": 0})

#counts = df["spam"].value_counts()
#print(f"Broj 1 (spam): {counts.get(1, 0)}")
#print(f"Broj 0 (ham): {counts.get(0, 0)}")

#df.to_csv("podaci_izmenjeni.csv", index=False)

#print("Završeno! Novi fajl je 'podaci_izmenjeni.csv'")

import pandas as pd

df = pd.read_csv("spam_Emails_data.csv")


df["spam"] = df["spam"].map({"Spam": 1, "Ham": 0})

counts = df["spam"].value_counts()
print(f"Broj 1 (spam): {counts.get(1, 0)}")
print(f"Broj 0 (ham): {counts.get(0, 0)}")

df.to_csv("novi1.csv", index=False)

print("Završeno! Novi fajl je 'novi1.csv'")