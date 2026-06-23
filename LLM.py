import openai
import os
from groq import Groq 
import panel as pn
from dotenv import load_dotenv
from pprint import pprint
import pandas as pd

def LLM_ans(Especie):


    load_dotenv()
    print(os.environ["GROQ_API_KEY"][:12])


    PROVIDER = "groq"
    groq_model = "llama-3.3-70b-versatile" 
    llm_client = Groq()
    model = groq_model
    df_eat = pd.read_csv("InfoEspecies.csv")
    mask = df_eat["label"] == Especie
    info = df_eat.Info.loc[mask].to_string()

    prompts = [{"role": "system", "content":'Eres un experto en alimentación humana, que tiene que dar inforamción sobre setas'},
            {"role": "system", "content": 'No es necesario que seas muy técnico explica todo de forma sencilla para todos'},
            {"role": "system", "content": 'usa emojis para hacerlo agradable a la vista'},
            {"role": "system", "content": 'Tampoco debes ofrecer más información de la que se te pida, eres parte de un programa con el que el usuario final no tiene la libertad de preguntarte nada directamente'}]
    if "comestible" in info:
        prompts+=[{"role": "system", "content":'Se te pasará una seta que se sabe que es comestible, queremos que nos des recetas de cocina'},
                {"role": "system", "content":'Se breve'},
                {"role": "system", "content":'Sigue la siguiente estructura: "Nombre: Aquí va el nombre de la seta\n Comestible: Si/NO\n Recetas de cocida", las recetas son una lista'},
                {"role": "user", "content":f'¿Qué puedes decirme sobre {Especie}?'}]
    elif "no comestible" in info:
        prompts+=[{"role": "system", "content":'Se te pasará una seta que se sabe que es no comestible, queremos que nos des recetas de cocina'},
                {"role": "system", "content":'Se breve'},
                {"role": "system", "content":'Sigue la siguiente estructura: "Nombre: Aquí va el nombre de la seta\n Comestible: Si/NO\n Razones para no consumirla", las razones dala como una lista '},
                {"role": "user", "content":f'¿Qué puedes decirme sobre {Especie}?'}]
    else:
        prompts+=[{"role": "system", "content":'En este caso no entra en ninguna categoría comestible o no comestible'},
                {"role": "user", "content":'Vale con que digas que no se sabe'}]

    respuesta_llm = llm_client.chat.completions.create(
        model=model,
        messages=prompts,
        temperature=0.2,
        max_tokens=1000
    )

    contenido_respuesta = respuesta_llm.choices[0].message.content
    print(contenido_respuesta)  


LLM_ans("Coprinellus micaceus")