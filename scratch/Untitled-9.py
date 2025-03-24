#%%
import spacy

nlp = spacy.load("en_core_web_sm")

text = "Ok my speed fusion connect only tops out at 40mbps on my two connections... both are 100mbps down here at my location and should seen about 125-150 down but I'm not getting over 40mbps... what is needed to see higher speeds? My router only  tops out at 300mbps but that's fast enough..."
text = "hello my name is Slim Shady"

doc = nlp(text)
print(doc)

#%%

token = doc[0]
# Print all properties of a spaCy Token
print("\nAll properties of a spaCy Token:")
for attr in dir(token):
    # Skip private attributes (those starting with '_')
    if not attr.startswith('_'):
        try:
            value = getattr(token, attr)
            # Check if it's a method or a property
            if not callable(value):
                print(f"{attr}: {value}")
        except Exception as e:
            print(f"{attr}: [Error: {e}]")

#%%

print(dir(token))

#%%

print(doc.vector)