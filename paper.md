# News-Research-Aid: From Summary to Perspective
### A Technical Engineering Paper

---

## 1. What I Wanted to Build

Political news is not just information — it is framed. The same event, reported by Al Jazeera, the BBC, and Fox News, will not simply use different words. It will foreground different causes, different victims, and different stakes. A reader who consumes only one source does not just get one perspective; they get one perspective that presents itself as neutral fact.

This problem is acute in contexts where accurate, multi-perspective news literacy matters most. In Model UN research, a delegate who understands only one country's framing of a conflict will struggle to anticipate objections, draft workable resolutions, or model the logic of opposing blocs. More broadly, any reader trying to form an independent opinion on a geopolitical event — the militarization of the Red Sea, nuclear non-proliferation negotiations, the use of private military companies — needs to understand not just what happened, but why different actors frame it the way they do, and where that framing comes from historically.

The tool I wanted to build was not another news summarizer. It was a perspective navigator: a system that could take a topic, surface current reporting from multiple sources, identify how each source frames the event differently, and then trace the historical roots of those competing framings. The most interesting version of this tool would give a reader a timeline — a visible record of how the same conflict or policy question has been framed over years — so that by the end, the reader could form their own opinion having been genuinely exposed to the full range of perspectives, not just one framing dressed up as objectivity.

The target user is anyone who needs to get seriously informed: a MUN delegate preparing a position paper, a student writing a policy brief, or a general reader tired of consuming news that sounds neutral but isn't.

---

## 2. The Rudimentary Baseline (Space 2)

The baseline I built before reaching Space 3 went through two versions.

**Space 1** established the simplest possible pipeline: call NewsAPI, retrieve the top five headlines on a topic, pass them to an AI model, and return a consolidated summary. This worked. The AI could compress five headlines into a coherent paragraph. But it had an obvious and fundamental problem: it was summarizing headlines, not articles, and it was treating all five sources as one unified voice. The output sounded authoritative. It was not.

**Space 2** improved on this by giving the user agency over source selection. Rather than summarizing five headlines automatically, the tool presented the top results and let the user choose one specific article to focus on. The AI then produced a much longer, more detailed summary of that single source — pulling out key claims, context, and implications. This was genuinely more useful than Space 1. A MUN delegate could pick the Reuters article, or the Al Jazeera article, and get a thorough briefing on what that specific source was saying.

What Space 2 could not do was tell the user anything about what they were missing. A detailed summary of a Reuters article tells you what Reuters said. It does not tell you that Reuters emphasized military escalation while Al Jazeera emphasized civilian casualties, or that this framing difference reflects a pattern that has been consistent across decades of coverage of the same region. The tool could produce an AI summary and gesture toward the concept of bias if prompted — but it had no mechanism for explaining where the bias came from, why it existed, or what the alternative framings looked like in practice.

The limitation of Space 2 was not a technical failure. The pipeline worked. The limitation was conceptual: summarizing one source more thoroughly does not help a reader understand the framing choices that source is making, or the history that produced those choices. A reader who finishes a Space 2 briefing knows more about one perspective. They do not know more about the landscape of perspectives — and they may, if anything, have more confidence in a single framing than is warranted.

That gap — between deeper summary and genuine perspective awareness — is what Space 3 is designed to address.
