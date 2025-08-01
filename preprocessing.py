import pandas as pd
from sklearn.model_selection import train_test_split


acc = pd.read_csv('accident.csv')
art = pd.read_csv('art.csv')
world = pd.read_csv('world.csv')
economy = pd.read_csv('economy.csv')
sport = pd.read_csv('sport.csv')

acc['label'] = 'accident'
art['label'] = 'art'
world['label'] = 'world'
economy['label'] = 'economy'
sport['label'] = 'sport'

concat = pd.concat([acc, art, world, economy, sport], ignore_index=True)

train_val, test = train_test_split(
    concat, 
    test_size=0.1, 
    random_state=42, 
    stratify=concat['label']
)

train, val = train_test_split(
    train_val, 
    test_size=0.1, 
    random_state=42, 
    stratify=train_val['label']
)

train.to_csv('train.csv', index=False)
val.to_csv('val.csv', index=False)
test.to_csv('test.csv', index=False)

