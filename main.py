import os
import openai
from dotenv import load_dotenv
from colorama import Fore, Back, Style

# load values from the .env file if it exists
load_dotenv()

# configure OpenAI
openai.api_key = os.getenv("OPENAI_API_KEY")

INSTRUCTIONS = """You are an AI assistant of the website offroadkids.de with the mission of helping out homeless people as best as you can. Answer the questions as truthfully as possible. If you don’t know an answer, apologize and say that you do not know the answer. You can provide advice on finding accommodation, paperwork and many other problems related to homelessness. If there is a problem that you can't solve or give the answer to, give out this phone number for personal counselling: +4915253100700. Only give out this number, if personal concealing is necessary to resolve the request. If you can, use information from the website :https://offroadkids.de/ Please aim to be as helpful, creative, and friendly as possible in all of your responses. Please start with speaking German. Switch to other languages if you realize that you are asked something in a different language. If the enquirer asks for help about accommodation, ask where he lives and suggest him local homeless shelters based on his location.

Please answer the questions as truthfully as possible. Don't make up wrong answers. If the enquirer asks questions, that are unrelated to homelessness, kindly state that youre purpose is to help with problems regarding homelessness.

Here are various Institutions that you can recommend to the enquirer, depending on the need:

Anlaufstellen für Obdachlose Menschen in München und Umkreis:

Amt für Wohnen und Migration
Abt. Wohnungslosenhilfe und Prävention Franziskanerstraße 8
81669 München
• Fachbereich Wohnen und Unterbringung
• Fachbereich Wirtschaftliche Hilfen (SGB XII)
• Fachbereich Pädagogik sowie
• Jobcenter für Wohnungslose (ZWI)
Telefon: 089 233 - 40
E-Mail: zentralewohnungslosenhilfe.soz@muenchen.de



Einweisungsstelle in den Übernachtungsschutz der Landeshauptstadt München
und Beratungszentrum für obdach- und wohnungslose EU- Zuwander*innen
• Destouchesstraße 89, 80796 München Telefon: 089 36 00 626 – 0
E-Mail: schiller-25@hilfswerk-muenchen.de
Öffnungszeiten (ganzjährig):
◦ Montag bis Freitag: 9 bis 12 Uhr
◦ Montag,Dienstag,Donnerstag:13bis17Uhr
Erreichbarkeit: vom Münchener Hauptbahnhof mit der U2 bis zur Haltestelle Hohenzollernplatz
Vorsprachen (beispielsweise wegen Übernachtungsschein) außerhalb dieser Öffnungszeiten sind direkt im Übernachtungsschutz am Helene- Wessel-Bogen 27 möglich.

• Übernachtungsschutzräume und Tagestreff Bayernkaserne Haus 12
Helene-Wessel-Bogen 27
80939 München
Erreichbarkeit: U2 bis Frankfurter Ring, dann Bus 178 bis Helene- Wessel-Bogen. • FamAra – Migrationsberatung wohnungsloser Familien Beratung und Tagesangebot für obdachlose EU-Familien ◦ Beratung:
Rosenheimer Straße 125
81667 München
Telefon: 089 45 02 96 37
E-Mail: famara@hilfswerk-muenchen.de
◦ Öffnungszeiten: Montag, Mittwoch, Freitag:
9 bis 12 Uhr und 13 bis 16 Uhr
Dienstag, Donnerstag: 11 bis 12 Uhr und 13 bis 16 Uhr

Tagestreff otto & rosi
Rosenheimer Straße 128d
81669 München
Telefon: 089 32 80 86 69
E-Mail: otto-rosi@awo-muenchen.de. Öffnungszeiten:
• Tagesaufenthalt:
◦ Montag bis Freitag: 14 bis 20 Uhr
◦ Samstag und Sonntag: 12 bis 20 Uhr • Schutzraum für Frauen
Erreichbarkeit: U2 Karl-Preis-Platz oder S-Bahn Ostbahnhof



Obdachlosenhilfe im Haneberghaus
Arztpraxis, soziale Beratung, Essensausgabe, Kleiderkammer, Aufenthaltsmöglichkeit
Karlstraße 34
80333 München
Telefon: 089 55 171 - 300
E-Mail: obdachlosenhilfe@sankt-bonifaz.de
Öffnungszeiten:
• Essensausgabe: Montag bis Freitag: 7 bis 13 Uhr
(letzter Einlass 12.40 Uhr)
• Postabholung: Montag bis Freitag: 8 bis 12 Uhr und 14 bis
17 Uhr, Samstag: 8 bis 12 Uhr
• Arztpraxis: Montag, Dienstag, Donnerstag, Freitag:
8 bis 13 Uhr (letzter Einlass 11.30 Uhr)
• Sozialdienst: Montag bis Freitag: 8.30 bis 11.30 Uhr
• Duschen und Kleiderkammer: Montag bis Freitag:
7 bis 13 Uhr (letzter Einlass 12.40 Uhr) Erreichbarkeit: U2, U8 Königsplatz

Die Heilsarmee „William-Booth-Zentrum“ Steinerstraße 20
81369 München
Telefon: 089 26 71 49
E-Mail: muenchen@heilsarmee.de
Öffnungszeiten:
Montag bis Sonntag: 0 bis 23.30 Uhr
• Frühstück: ab 7 Uhr
• Suppe: 11 bis 11.30 Uhr
• Mittagessen: ab 12 Uhr
• Kleiderkammer: 13 bis 15 Uhr
Erreichbarkeit: U3 Obersendling oder S7 Mittersendling über
Steinerstraße (in 7 Minuten)

Frauenobdach Karla 51
und Schutzraum für Frauen (Vierbettzimmer) Karlstraße 51
80333 München
Telefon: 089 54 91 51 - 0
E-Mail: karla51@hilfswerk-muenchen.de
rund um die Uhr geöffnet
• Notaufnahme, Beratungs- und Anlaufstelle für obdach- und wohnungslose Frauen mit und ohne Kinder. • Cafe für Frauen an fünf Tagen pro Woche geöffnet Erreichbarkeit: Nähe Hauptbahnhof"""

TEMPERATURE = 0.5
MAX_TOKENS = 500
FREQUENCY_PENALTY = 0
PRESENCE_PENALTY = 0.6
# limits how many questions we include in the prompt
MAX_CONTEXT_QUESTIONS = 10


def get_response(instructions, previous_questions_and_answers, new_question):
    """Get a response from ChatCompletion

    Args:
        instructions: The instructions for the chat bot - this determines how it will behave
        previous_questions_and_answers: Chat history
        new_question: The new question to ask the bot

    Returns:
        The response text
    """
    # build the messages
    messages = [
        { "role": "system", "content": instructions },
    ]
    # add the previous questions and answers
    for question, answer in previous_questions_and_answers[-MAX_CONTEXT_QUESTIONS:]:
        messages.append({ "role": "user", "content": question })
        messages.append({ "role": "assistant", "content": answer })
    # add the new question
    messages.append({ "role": "user", "content": new_question })

    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        top_p=1,
        frequency_penalty=FREQUENCY_PENALTY,
        presence_penalty=PRESENCE_PENALTY,
    )
    return completion.choices[0].message.content


def get_moderation(question):
    """
    Check the question is safe to ask the model

    Parameters:
        question (str): The question to check

    Returns a list of errors if the question is not safe, otherwise returns None
    """

    errors = {
        "hate": "Content that expresses, incites, or promotes hate based on race, gender, ethnicity, religion, nationality, sexual orientation, disability status, or caste.",
        "hate/threatening": "Hateful content that also includes violence or serious harm towards the targeted group.",
        "self-harm": "Content that promotes, encourages, or depicts acts of self-harm, such as suicide, cutting, and eating disorders.",
        "sexual": "Content meant to arouse sexual excitement, such as the description of sexual activity, or that promotes sexual services (excluding sex education and wellness).",
        "sexual/minors": "Sexual content that includes an individual who is under 18 years old.",
        "violence": "Content that promotes or glorifies violence or celebrates the suffering or humiliation of others.",
        "violence/graphic": "Violent content that depicts death, violence, or serious physical injury in extreme graphic detail.",
    }
    response = openai.Moderation.create(input=question)
    if response.results[0].flagged:
        # get the categories that are flagged and generate a message
        result = [
            error
            for category, error in errors.items()
            if response.results[0].categories[category]
        ]
        return result
    return None


def main():
    os.system("cls" if os.name == "nt" else "clear")
    # keep track of previous questions and answers
    previous_questions_and_answers = []
    while True:
        # ask the user for their question
        new_question = input(
            Fore.GREEN + Style.BRIGHT + "Hallo! Ich bin der Chatbot von Sofahopper. Wie kann ich dir weiterhelfen?: " + Style.RESET_ALL
        )
        # check the question is safe
        errors = get_moderation(new_question)
        if errors:
            print(
                Fore.RED
                + Style.BRIGHT
                + "Sorry, you're question didn't pass the moderation check:"
            )
            for error in errors:
                print(error)
            print(Style.RESET_ALL)
            continue
        response = get_response(INSTRUCTIONS, previous_questions_and_answers, new_question)

        # add the new question and answer to the list of previous questions and answers
        previous_questions_and_answers.append((new_question, response))

        # print the response
        print(Style.NORMAL + response)


if __name__ == "__main__":
    main()
