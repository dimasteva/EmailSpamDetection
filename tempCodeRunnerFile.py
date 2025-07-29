df = pd.read_csv('emails.csv')

# Pretpostavljamo da je 'spam' kolona sa vrednostima 0 (nije spam) i 1 (spam)
# Prvo izdvajamo većinsku i manjinsku klasu
df_majority = df[df['spam'] == df['spam'].value_counts().idxmax()]
df_minority = df[df['spam'] == df['spam'].value_counts().idxmin()]

# Downsampling većinske klase
df_majority_downsampled = df_majority.sample(n=len(df_minority), random_state=42)

# Kombinujemo nazad
df_downsampled = pd.concat([df_majority_downsampled, df_minority])

# Promešamo redosled
df_downsampled = df_downsampled.sample(frac=1, random_state=42).reset_index(drop=True)

print(df_downsampled['spam'].value_counts())