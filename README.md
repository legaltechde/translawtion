# Transla{w}tion

![Transla{w}tion – Screenshot](images/screenshot.png)

**Echtzeit-Übersetzung von Gebärdensprache für Behördengänge**

Transla{w}tion ist ein Prototyp für eine Software, die Menschen mit Gebärdensprache bei Behördengängen unterstützen soll. Ziel der Idee ist es, die Kommunikation zwischen gebärdensprachlichen und lautsprachlichen Personen direkt über einen Laptop zu ermöglichen und dadurch in geeigneten Situationen die Abhängigkeit von einem zusätzlichen Dolmetscher zu reduzieren.

Die App soll dabei eine Alternative zur ausschließlich textbasierten Kommunikation für Gebärdensprachler darstellen. Sie sollen nicht nur über E-Mail oder mit Unterstützung eines Dolmetschers kommunizieren können, sondern auch selbstständig und direkt ein Gespräch mit ihrem Gegenüber führen können.

> **Hinweis:** Transla{w}tion ist ein Hackathon-Prototyp und kein fertiges oder für den produktiven Einsatz freigegebenes Übersetzungssystem.

## Entstehung

Die App ist im Rahmen des **Bucerius Law School Hackathon 2025** im **Mai 2025** entstanden.

Der Prototyp wurde entwickelt, um eine mögliche technische Lösung für barriereärmere Kommunikation im Kontext von Behörden zu zeigen. Dabei stand insbesondere die Idee im Vordergrund, Gebärdensprachkommunikation und normale Sprache in einem gemeinsamen Gesprächsfenster zusammenzuführen.

## Versionen und Systemvoraussetzungen

Im Repository befinden sich zwei Windows-Varianten:

- **Windows mit MediaPipe:** Version mit MediaPipe zur Unterstützung der Gebärdenerkennung. Diese Variante funktioniert **nur mit Python 3.11** (Stand: **09.08.2026**).
- **Windows und macOS ohne MediaPipe:** Eine Variante ohne MediaPipe, die sowohl unter **Windows als auch macOS** verwendet werden kann.

Die MediaPipe-Version ist damit insbesondere hinsichtlich der Python-Version eingeschränkt. Die Angaben entsprechen dem Stand des Projekts vom **09.08.2026**.

## Funktionsweise

Das grundlegende Konzept besteht aus zwei Kommunikationswegen:

- **Gebärdensprache → Text:** Eine Kamera erfasst die gebärdende Person. Einzelne erkannte Gebärden werden fortlaufend als Wörter angezeigt.
- **KI-Vorschlag:** Aus den fortlaufend erkannten Wörtern wird im Hintergrund ein sinnvoller deutscher Satz vorgeschlagen.
- **Bestätigung durch die gebärdende Person:** Der vorgeschlagene Satz kann angenommen oder abgelehnt werden. So soll die gebärdende Person kontrollieren können, ob die Software ihre Aussage richtig verstanden und formuliert hat.
- **Normale Sprache → Gespräch:** Eine lautsprachlich kommunizierende Person kann im selben Chat antworten. Dadurch soll ein gemeinsames Gespräch zwischen gebärdensprachlichen und lautsprachlichen Personen möglich werden.
- **Sprachausgabe:** Übersetzte bzw. bestätigte Inhalte können zusätzlich über eine Sprachausgabe ausgegeben werden.

Vereinfacht lässt sich der Ablauf so darstellen:

```text
Gebärde
   ↓
Kamera / Erkennung
   ↓
fortlaufend erkannte Wörter
   ↓
KI formuliert Satzvorschlag
   ↓
Person bestätigt oder lehnt ab
   ↓
gemeinsamer Chat
   ↕
lautsprachliche Antwort
```

Die Oberfläche ist dabei bewusst als einfache Desktop-Anwendung konzipiert. Sie soll grundsätzlich auf einem gewöhnlichen Laptop mit Kamera eingesetzt werden können.

## Beispiel für einen Gesprächsablauf

Ein möglicher Behördengang könnte beispielsweise so aussehen:

```text
10:15 – Person B (Sprache): Guten Tag, wie kann ich Ihnen helfen?

10:15 – System (Erkannt): ich | möchte | einen | ausweis | beantragen

10:15 – System (Vorschlag): Ich möchte einen Ausweis beantragen.

10:15 – Person A (Gebärdensprache): ✓ Übernehmen

10:15 – System: Übersetzung gesendet.

10:16 – Person B (Sprache): Gerne. Benötigen Sie einen Personalausweis oder einen Reisepass?

10:16 – System (Erkannt): personalausweis

10:16 – System (Vorschlag): Ich möchte einen Personalausweis beantragen.

10:16 – Person A (Gebärdensprache): ✓ Übernehmen

10:16 – System: Übersetzung gesendet.

10:17 – Person B (Sprache): Dafür benötige ich bitte Ihren bisherigen Ausweis
und gegebenenfalls ein aktuelles biometrisches Passfoto.
```

Der entscheidende Punkt ist dabei, dass **die erkannte Wortfolge nicht automatisch als endgültige Aussage versendet werden muss**. Erst der Satzvorschlag der KI kann von der gebärdenden Person bestätigt oder abgelehnt werden.

## Status des Projekts

Die App wurde als Hackathon-Prototyp entwickelt und **nicht weiterentwickelt**.

Eine Weiterentwicklung ist **aktuell ebenfalls nicht geplant**. Es bestehen derzeit **keine Pläne**, das Projekt zu einem produktiven oder dauerhaft gepflegten System auszubauen.

Der hier veröffentlichte Stand dient daher in erster Linie als Dokumentation und als Einblick in die im Hackathon entwickelte Idee und den entstandenen Prototyp.

## Wichtige Einschränkungen

Der Prototyp ist **nicht als verlässliches Übersetzungs- oder Kommunikationssystem für reale Behördengänge gedacht**. Insbesondere können automatische Gebärdenspracherkennung und KI-basierte Sprachformulierung Fehler machen.

Bei einer tatsächlichen Anwendung in rechtlich oder administrativ relevanten Situationen müssten unter anderem Genauigkeit, Datenschutz, Barrierefreiheit, Sicherheit, Einwilligung, Haftungsfragen und die zuverlässige Kommunikation mitgedacht und umfangreich geprüft werden.

Die im Prototyp dargestellte Idee ist daher als technisches Konzept und nicht als Ersatz für professionelle Gebärdensprachdolmetschung oder andere notwendige Kommunikationshilfen zu verstehen.

## Kontakt

Bei Interesse am Projekt, dem Prototyp oder der dahinterstehenden Idee kann Kontakt aufgenommen werden:

**legaltechde@mail.de**

---

*Entstanden beim Bucerius Law School Hackathon 2025 · Mai 2025*
