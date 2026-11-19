to run public url:

run lable studio in one terminal with:
$env:LABEL_STUDIO_BASE_DATA_DIR = "C:\labelstudio_validation"                                                                                 
$env:LABEL_STUDIO_HOST = "https://chain-marshy-postbox.ngrok-free.dev"
$env:CSRF_TRUSTED_ORIGINS = "https://chain-marshy-postbox.ngrok-free.dev"
$env:LABEL_STUDIO_DISABLE_SIGNUP_WITHOUT_LINK = "true"                                                          
.\.venv-labelstudio\Scripts\label-studio.exe start --port 8081

and run in another terminal:
ngrok http 8081 (if it doesnt work configure the authentification before, look in proton pass)

to run local admin project:
$env:LABEL_STUDIO_BASE_DATA_DIR = "C:\Users\lucia\OneDrive\Documents\CognitiveSystems\Seoul\PM\new\data\labelstudio_batches"                  .\.venv-labelstudio\Scripts\label-studio.exe start      (maybe add port 8080)


Instructions for labelling:

Vielen Dank nochmal!! Um eine Übersicht zu geben, was jede Kategorie bedeutet, habe ich ein kleines "Regelbuch" zusammengestellt: 

Es geht darum ein NLI-Modell zu Fine-Tunen (also ein ML-Modell besser und spezifischer zu machen). Hierbei geht es um den Zusammenhang von Stance und Sentiment, daher sind das ie beiden Faktoren, die wichtig sind beim labeln. Hier brauche ich eure Hilfe. Da es dann doch oft relativ subjektiv ist, wie negativ oder positiv etwas gesagtes ist, brauche ich in der Wisschenschaftlichen Arbeit ein paar mehr Menschen, die miene Ergebnisse unabhängig voneinander Bestätigen.

ICh werde dam Modell die gleiche Hypothese geben:
„Migration nach Deutschland soll reduziert oder restriktiver gehandhabt werden.“

Mit dem Thema Migration ist alles rundum Grenze/Zugang, Asylverfahren, Abschiebung/Aufenthalt, Familiennachzug, und Sozialleistungen als Steuerungsmittel gemeint.


1. Stance: Stimmt der Text der Hypothese zu, oder nicht?
Hier geht es vorallem um ein JA (Pro), oder nicht JA (Con). Con heißt hier nicht unbedingt das genaue Gegenteil. Das kann natürlcih auch teilweise der Fall sien, dass mehr EInwanderung gefordert wird, aber Con kann auch schon ein ablehnen der Begrenzung sein.

pro: Fordert weniger oder restriktivere Migration, bewertet die Menge als zu hoch, oder unterstützt eine restriktive Maßnahme (Abschiebung, Zurückweisung, Aussetzung des Familiennachzugs …).
„Es kommen zu viele.“ – „Abgelehnte müssen abgeschoben werden.“

contra:	Fordert mehr Aufnahme oder keine Begrenzung, lehnt eine restriktive Maßnahme ab, weil sie Schutz/Zugang verweigert oder Menschen schadet, oder fordert eine liberale Maßnahme.
„Deutschland braucht Zuwanderung.“ – „Familien müssen nachkommen dürfen.“
neutral	Keine Forderung oder Mengenbewertung, oder gemischt (siehe unten).

Was ist alles Neutraler Stance:

    Kritik oder Abwertung (Kriminalität, Kosten, Politiker) OHNE Forderung
    Lob von Migrant:innen OHNE Forderung
    Gemischt: z. B. „mehr Fachkräfte, weniger irreguläre Migration“
    Nur wiedergegebene Positionen anderer
    Ablehnung nur aus rechtlichen oder praktischen Gründen („verfassungswidrig“, „wirkt nicht“)
    Humanitäre Sorge OHNE Forderung nach Aufnahme
    Nur „Migration ordnen/steuern“, Integration, Einbürgerung, reine Fakten
    „Fluchtursachen bekämpfen“, außer es heißt ausdrücklich: damit weniger kommen (dann pro)

Test bei Unsicherheit: Passt ein gegenteiliger Satz ohne Widerspruch dahinter? Dann neutral.


2. Tone: Wie ist der Text formuliert?

Nur die Sprache zählt, nicht der Stance der Person, oder das Thema.
negativ: alles was abwertend, angreifend, empört, spöttisch, dramatisierend formuliert ist. „Chaos“, „Schande“, „Versagen“
neutral: alles was sachlich, beschreibend, argumentierend formuliert ist. Auch harte Forderungen und sachliche Kritik. Solange das Narrativ neutral bleibt.
positiv: alles was anerkennend, lobend, dankend, zuversichtlich formuliert ist. „stolz“, „Erfolg“, „großartig“
gemischt: wenn deutlich positive und deutlich negative Teile mit ähnlichem Gewicht vorkommen. Hier bitte vorsichtig sein.

Trauriges Thema ist nicht gleich negativ. Ein einzelnes Wort entscheidet nicht, sondern der Gesamteindruck.

Falls du dir unsicher bist, versuche die intuitivste Option zu wählen und schreibe eine kurze Notiz ins Kommentarfeld. Bei technischen oder inhaltlichen Fargen: bitte bei mir melden :) 

Bleibe nicht zu lange an einem Chunk hängen, sollte nicht länger als 2 Minuten pro Chunk dauern. Wenn doch, skipp einfach xx


