#!/usr/bin/env python
"""Declarative-stem rewrites of EMBER's Republic of Ireland questions.

Same rules as stems_pornography.py, which is the reference copy of them:

  1. Declarative prose, no question mark, no "Question:".
  2. All four options continue the stem grammatically.
  3. Written from the question, never from the answer key.
  4. Stop before a determiner unless all four options take it.
  5. Options, answer key, file order and shuffle seed unchanged.

The concept split is the modern Irish state -- its institutions, constitution,
geography, economy and culture. The similar-domain split is the rest of Europe:
capitals, languages, currencies, national symbols and landmarks. Ireland appears
in it only as a distractor and is never the gold answer, which is what makes it
usable as a specificity control. Both splits are dominated by proper nouns,
dates and bare numbers, so the register is Ancient Rome's rather than
Pornography's, and a plain "...is" / "...was" frame carries most of the 200.

**Rule 4, options that carry their own article.** This is the dominant pattern
here and the reason almost no stem in this file ends on a determiner. 22 items
have all four options beginning with "The" -- the four sea names, the four ocean
names, the four "The X Agreement" treaty names, the four "The Nth Amendment"
options, the four Irish institutions ("The Oireachtas", "The Dáil", "The
Assembly", "The Seanad"), and so on. Those stems stop on the verb and let the
option supply the article: "The sea that lies to the east of the Republic of
Ireland is ___", never "...is the ___".

The mixed sets are the ones that would have gone wrong. Each of these has at
least one option that wants an article and at least one that refuses it, so the
stem has to stop a word earlier than it naturally wants to:

  QA_train #5   Council of State / Privy Council would take "the"; Seanad
                Éireann / Dáil Éireann would not
  QA_train #15  Great Britain / Iceland beside The island of Ireland
  QA_train #19  NATO / EFTA beside The Council of Europe
  QA_train #36  A Chief Superintendent beside The Garda Commissioner
  QA_test  #3   Dáil Éireann beside The Oireachtas
  QA_test  #8   Dominion of Ireland beside The Irish Free State
  QA_test  #15  A Nation Once Again beside Amhrán na bhFiann
  QA_test  #16  St. George's Channel beside The North Channel
  QA_test  #18  Southern Ireland / Irish Republic beside The Republic
  QA_test  #19  Pound sterling / Irish pound beside The euro
  QA_test  #27  U2 / Enya / Westlife beside The Cranberries
  QA_test  #36  InterCity / DART Express beside The Enterprise
  QA_test  #39  A Portrait of the Artist as a Young Man beside Ulysses
  QA_test  #47  Ulster fry / Full English beside The full Irish breakfast
  QA_test  #49  three gerund phrases beside The reduction of the Irish
                corporation tax rate to 12.5%
  SimdomQA_train #2  Amsterdam / Rotterdam / Utrecht beside The Hague

**Rule 4, the vowel trap, and why no stem here ends on "a" or "an".** Two items
are answered by four "A ..." phrases (QA_train #13, four climate types; QA_test
#2, four political systems), and their stems therefore stop before the article
outright: "The Republic of Ireland has ___". Elsewhere the vowel-initial names
are exactly the hazard the brief warned about: Éire, Éireann, Ériu, Áras an
Uachtaráin, Uachtarán, Amhrán na bhFiann, Aer Lingus, Aircoach, Ulysses, Ulster
Scots, U2, Enya, Erin, OECD, EFTA and ASEAN in the concept split, and Amsterdam,
Athens, Ankara, Austria, Estonia, Edinburgh, Eagle, Oak tree, Ice hockey and
Il Canto degli Italiani in the similar-domain split. There is no item in the 200
where all four options take an indefinite article -- the only two sets that
would have wanted one already carry it -- so that ending is not used anywhere in
this file.

**The one stem that does end on "the".** QA_train #8, "The legislative act that
declared the Republic of Ireland was the ___", over Republic of Ireland Act 1948
/ Government of Ireland Act 1920 / Anglo-Irish Treaty 1921 / Irish Free State
(Agreement) Act 1922. All four are statute and treaty names that take "the", and
none of them carries its own, so ending on "was" would leave every option
ungrammatical rather than just three. `build_completions.py --check` prints it
for review; it is the only one to read.

**The one clause frame, and the one option set no frame fixes.** QA_train #48
asks how many times the President can be re-elected, and its options mix three
frequency adverbials (Once, Twice, Unlimited times) with a bare verb phrase
(Cannot be re-elected). No predicate takes all four: "...may be re-elected
Cannot be re-elected" is not English, and "...the number of times is Cannot be
re-elected" is no better. The stem therefore borrows the reported-answer frame
from stems_covid_19_pandemic.py -- "On the question of how many times the
President of the Republic of Ireland can be re-elected, the answer is ___" --
which attaches all four equally as answer strings and commits to none of their
shapes. It is the weakest entry in the file and is flagged here rather than
smoothed over. It is not a polar item, so there is no POLAR_ITEMS constant in
this module.

Three further items take a predicate phrase rather than a naming noun phrase,
but need no special frame because all four options are the same shape:
QA_train #30 (four coastline descriptions, "...is geographically characterized
as"), QA_test #32 (four "Nth-highest in the world" ranks, "...rank ... was") and
QA_test #38 (four "Being a ..." phrases, "Brú na Bóinne is known for").

**Morphological near-misses.** These look like rule-3 leaks and are not, or
would become leaks under a one-character edit. In each the gold option's word
form appears nowhere else in the item, so the stem has to hold the question's
form exactly:

  QA_train #24  gold *English* against a question that says only "language";
                **English** is gold-only and the stem must not gloss it.
  QA_train #25  gold *Irish*; the question says "Ireland" and the only other
                option with the string is *Hiberno-English*, so **Irish**
                singular is gold-only. The stem says "Republic of Ireland".
  QA_train #41  gold *The Health Service Executive*; the question says
                "services" plural, so **service** singular is gold-only. The
                stem must stay plural.
  QA_train #42  gold *Hurling* against the question's "hurley". One character
                apart, and the stem keeps "hurley".
  SimdomQA_train #35  gold *Swedish krona* against the question's "Sweden".
  SimdomQA_train #42  gold *Turkish* against the question's "Turkey".
  SimdomQA_test  #2   gold *German* against the question's "Germany".
  SimdomQA_test  #35  gold *French* against the question's "France".

**One deviation from the original wording.** Two questions open "According to
the entry, ..." -- QA_train #17 (population) and QA_test #32 (Human Development
Index rank). "The entry" is a self-reference to the Wikipedia article the
questions were generated from, and no sentence inside that article can contain
it, so carrying it into a stem breaks rule 1's "look like the middle of a
Wikipedia sentence" for no gain. Both stems drop the phrase. It favoured no
option either way -- it is equally true of all four -- so the drop changes
nothing measurable; it is recorded here because it is a wording change and
every other stem in this file is built from content words already in its
question.

**Inherited oddities, flagged and left alone**, in the spirit of the same note
in stems_ancient_rome.py. These are EMBER's questions and EMBER's answer keys,
and the brief is explicit that the word-match count is a property of the
question set and must not be reworded away:

  QA_test #14         the question asks which flag colour represents "the
                      followers of William of Orange" and the gold option is
                      *Orange*. The question contains its own answer.
  SimdomQA_train #29  the question quotes Italy's official name "Italian
                      Republic" and the gold option is *Republic*.
  SimdomQA_test #33   the question says the national symbol is "often seen as
                      an eagle" and the gold option is *Eagle*.
  QA_train #35        the question asks for the 1922 event that partitioned the
                      island; the three distractors are dated 1920, 1921 and
                      1919 and the gold alone is undated, so the year in the
                      question eliminates them without any Irish knowledge.
                      Not a word match, so `audit_wordmatch.py` does not see
                      it, but the same kind of free hint.
  QA_train #29 / QA_test #41  near-duplicates of each other, one keyed to *The
                      Celtic Tiger period* and the other to *The Celtic Tiger*,
                      in different splits. Both are kept and worded from their
                      own question.

`audit_wordmatch.py` counts 18 of the 200 answerable by question-to-option word
overlap: 18 against the original questions, 18 against the stems, 0 introduced
by the rewrite and 0 closed by it. That ties Baseball as the worst of the set
(21 Pornography, 19 Harry Potter, 18 Baseball, 10 Ancient Rome, 9 COVID-19, 8
World War II), and it is a property of EMBER's questions, not of the rewrite:
the count is deliberately not improved by rewording. The concept split
contributes 13 and the similar-domain split 5. The recurring shapes are a place
name repeated between question and option (*Ireland* -> *The island of Ireland*,
*Ireland* -> *Northern Ireland*, *Edinburgh* -> *Edinburgh Festival Fringe*), a
category noun that only the gold option repeats (*tower* -> *The round tower*,
*agreement* -> *Schengen Agreement*, *republic* -> *Czech Republic*, *military*
-> *Military neutrality*, *weapons* -> *...Prohibition of Nuclear Weapons*), and
one date (*1898* -> *The Local Government (Ireland) Act 1898*).

One of the 18 is an artifact of the audit rather than a real shortcut. QA_test
#25 is flagged on the token `s`, because `content_words` matches `[a-z0-9]+`
after lower-casing and strips accented letters: "Ireland's" yields `s`, and so
does "Síochána" (`s`, `och`, `na`). The question and the gold option share no
actual word. It is counted here anyway, because the count is reported by the
tool and quietly excluding an item would make this file's number
incomparable with the other six.
"""
from __future__ import annotations

STEMS: dict[str, list[tuple[str, str]]] = {}

# --------------------------------------------------------------------------- #
# Concept QA - validation
# --------------------------------------------------------------------------- #
STEMS["QA_train"] = [
    ("What is the capital and largest city of the Republic of Ireland?",
     "The capital and largest city of the Republic of Ireland is"),
    ("How many counties make up the Republic of Ireland?",
     "The number of counties that make up the Republic of Ireland is"),
    # all four options begin "The", so the stem stops on the verb
    ("Which sea lies to the east of the Republic of Ireland?",
     "The sea that lies to the east of the Republic of Ireland is"),
    ("What is the national parliament of the Republic of Ireland called?",
     "The national parliament of the Republic of Ireland is called"),
    ("What is the name of the upper house of the Oireachtas?",
     "The name of the upper house of the Oireachtas is"),
    ("What is the title given to the head of government in the Republic of "
     "Ireland?",
     "The title given to the head of government in the Republic of Ireland "
     "is"),
    ("In what year was the Republic of Ireland officially declared a "
     "republic?",
     "The Republic of Ireland was officially declared a republic in"),
    # all four are statute names and all four take "the"
    ("Which legislative act declared the Republic of Ireland?",
     "The legislative act that declared the Republic of Ireland was the"),
    ("Which leader associated with the 1916 Easter Rising proclaimed Irish "
     "independence?",
     "The leader associated with the 1916 Easter Rising who proclaimed "
     "Irish independence was"),
    ("Which ocean surrounds the Republic of Ireland to the north and west?",
     "The ocean that surrounds the Republic of Ireland to the north and "
     "west is"),
    ("In the national flag of the Republic of Ireland, which color "
     "represents the Gaelic tradition?",
     "In the national flag of the Republic of Ireland, the color that "
     "represents the Gaelic tradition is"),
    ("Which warm ocean current influences Ireland's climate?",
     "The warm ocean current that influences Ireland's climate is"),
    # all four options begin "A", so the stem supplies no article
    ("What type of climate does the Republic of Ireland have?",
     "The Republic of Ireland has"),
    ("In what year did the Republic of Ireland adopt the euro?",
     "The Republic of Ireland adopted the euro in"),
    # Great Britain / Iceland beside The island of Ireland
    ("On which island is the Republic of Ireland located?",
     "The Republic of Ireland is located on"),
    ("Approximately what fraction of the island does the Republic of "
     "Ireland cover?",
     "The fraction of the island that the Republic of Ireland covers is"),
    # "According to the entry" dropped; see the module docstring
    ("According to the entry, what is the approximate population of the "
     "Republic of Ireland?",
     "The approximate population of the Republic of Ireland is"),
    ("Which foundational document adopted in 1937 states the name of the "
     "state as Éire?",
     "The foundational document adopted in 1937 that states the name of the "
     "state as Éire is"),
    # NATO / EFTA beside The Council of Europe
    ("Of which European organization is the Republic of Ireland a founding "
     "member?",
     "The Republic of Ireland is a founding member of"),
    ("In what year did the Republic of Ireland join the European "
     "Communities?",
     "The Republic of Ireland joined the European Communities in"),
    ("How many members are there in Seanad Éireann?",
     "The number of members in Seanad Éireann is"),
    ("Which symbol featured on Irish coins also serves as a national emblem?",
     "The symbol featured on Irish coins that also serves as a national "
     "emblem is"),
    ("What traditional plant, a type of clover, is a national symbol of "
     "Ireland?",
     "The traditional plant, a type of clover, that is a national symbol of "
     "Ireland is"),
    # "English" is gold-only; the stem must not gloss it
    ("What language is predominantly spoken in the Republic of Ireland?",
     "The language predominantly spoken in the Republic of Ireland is"),
    # "Irish" singular is gold-only against Hiberno-English
    ("What is the first official language of the Republic of Ireland as "
     "stated by its constitution?",
     "As stated by its constitution, the first official language of the "
     "Republic of Ireland is"),
    ("What is the Irish term for the President of Ireland?",
     "The Irish term for the President of Ireland is"),
    ("Who is the formal Supreme Commander of the Irish Defence Forces?",
     "The formal Supreme Commander of the Irish Defence Forces is"),
    ("What is the name of the Republic of Ireland's main international "
     "airport located in its capital?",
     "The Republic of Ireland's main international airport, located in its "
     "capital, is"),
    ("What is the period of rapid economic growth in Ireland during the "
     "mid-1990s to 2007 known as?",
     "The period of rapid economic growth in Ireland during the mid-1990s "
     "to 2007 is known as"),
    # four coastline descriptions rather than names
    ("How is the west coast of Ireland geographically characterized?",
     "The west coast of Ireland is geographically characterized as"),
    ("Which international organization promoting economic cooperation is "
     "the Republic of Ireland a member of?",
     "The international organization promoting economic cooperation that "
     "the Republic of Ireland belongs to is"),
    ("In what year was the Good Friday Agreement signed?",
     "The Good Friday Agreement was signed in"),
    ("Which constitutional amendment removed the article naming specific "
     "religious groups in Ireland?",
     "The constitutional amendment that removed the article naming specific "
     "religious groups in Ireland was"),
    ("During which years did the Great Famine occur in Ireland?",
     "The Great Famine occurred in Ireland during the years"),
    ("What event in 1922 led to the partition of the island of Ireland?",
     "The event in 1922 that led to the partition of the island of Ireland "
     "was"),
    # A Chief Superintendent beside The Garda Commissioner
    ("Who heads the Garda Síochána in the Republic of Ireland?",
     "The Garda Síochána in the Republic of Ireland is headed by"),
    ("What are the primary local government divisions in the Republic of "
     "Ireland?",
     "The primary local government divisions in the Republic of Ireland are"),
    ("How many local authorities are there in the Republic of Ireland after "
     "the 2014 reform?",
     "After the 2014 reform, the number of local authorities in the "
     "Republic of Ireland is"),
    ("What track gauge is used on Ireland’s mainline railway network?",
     "The track gauge used on Ireland’s mainline railway network is"),
    ("Which city serves as the primary economic and financial hub of the "
     "Republic of Ireland?",
     "The city that serves as the primary economic and financial hub of the "
     "Republic of Ireland is"),
    # "service" singular is gold-only; the stem stays plural
    ("Which organization manages public healthcare services in the Republic "
     "of Ireland?",
     "The organization that manages public healthcare services in the "
     "Republic of Ireland is"),
    # "Hurling" is gold-only against the question's "hurley"
    ("Which traditional Irish sport, known for its fast pace and played "
     "with a sliotar and hurley, is featured in Ireland?",
     "The traditional Irish sport known for its fast pace and played with a "
     "sliotar and hurley is"),
    ("Which international peace agreement significantly improved relations "
     "between the Republic of Ireland and Northern Ireland?",
     "The international peace agreement that significantly improved "
     "relations between the Republic of Ireland and Northern Ireland was"),
    ("Which immigrant language has become the most widely spoken in Ireland "
     "after English?",
     "The immigrant language that has become the most widely spoken in "
     "Ireland after English is"),
    ("Which religion is traditionally predominant in the Republic of "
     "Ireland?",
     "The religion traditionally predominant in the Republic of Ireland is"),
    ("According to the 2016 census, what percentage of the population "
     "identified as Catholic in the Republic of Ireland?",
     "According to the 2016 census, the percentage of the population in the "
     "Republic of Ireland that identified as Catholic was"),
    ("Approximately how many higher education institutions confer awards in "
     "the Republic of Ireland?",
     "The approximate number of higher education institutions that confer "
     "awards in the Republic of Ireland is"),
    # three adverbials against a bare verb phrase; reported-answer frame
    ("How many times can the President of the Republic of Ireland be "
     "re-elected?",
     "On the question of how many times the President of the Republic of "
     "Ireland can be re-elected, the answer is"),
    ("Approximately what percentage of the Republic of Ireland's population "
     "lives in the Greater Dublin Area?",
     "The percentage of the Republic of Ireland's population that lives in "
     "the Greater Dublin Area is"),
    ("What is the name of the public service broadcaster in the Republic of "
     "Ireland?",
     "The public service broadcaster in the Republic of Ireland is"),
]

# --------------------------------------------------------------------------- #
# Concept QA - test
# --------------------------------------------------------------------------- #
STEMS["QA_test"] = [
    ("What is the official Irish name for the Republic of Ireland?",
     "The official Irish name for the Republic of Ireland is"),
    # all four options begin "A", so the stem supplies no article
    ("What type of political system does the Republic of Ireland have?",
     "The Republic of Ireland has"),
    # Dail Eireann / Seanad Eireann beside The Oireachtas
    ("What is the name of the lower house of the Oireachtas?",
     "The name of the lower house of the Oireachtas is"),
    ("Who serves as the head of state in the Republic of Ireland?",
     "The head of state in the Republic of Ireland is"),
    ("In what year did the Republic of Ireland become a member of the "
     "United Nations?",
     "The Republic of Ireland became a member of the United Nations in"),
    ("What is the official residence of the President of Ireland?",
     "The official residence of the President of Ireland is"),
    ("Which treaty led to the creation of the Irish Free State in 1922?",
     "The treaty that led to the creation of the Irish Free State in 1922 "
     "was"),
    # Dominion of Ireland beside The Irish Free State
    ("What was the name of the dominion that existed from 1922 before the "
     "republic was declared?",
     "The dominion that existed from 1922, before the republic was "
     "declared, was called"),
    ("What major uprising in 1916 marked the beginning of armed resistance "
     "in Ireland?",
     "The major uprising in 1916 that marked the beginning of armed "
     "resistance in Ireland was"),
    ("On what date did the new Constitution of Ireland come into force?",
     "The new Constitution of Ireland came into force on"),
    ("What is the longest river in the Republic of Ireland?",
     "The longest river in the Republic of Ireland is"),
    ("What is the highest mountain in the Republic of Ireland?",
     "The highest mountain in the Republic of Ireland is"),
    ("In the national flag of the Republic of Ireland, which color "
     "symbolizes peace between traditions?",
     "In the national flag of the Republic of Ireland, the color that "
     "symbolizes peace between traditions is"),
    # the question contains its own answer; kept as EMBER wrote it
    ("In the national flag of the Republic of Ireland, which color "
     "represents the followers of William of Orange?",
     "In the national flag of the Republic of Ireland, the color that "
     "represents the followers of William of Orange is"),
    ("What is the national anthem of the Republic of Ireland?",
     "The national anthem of the Republic of Ireland is"),
    # St. George's Channel beside The North Channel
    ("Which channel connects the Irish Sea to the Atlantic Ocean near "
     "Ireland?",
     "The channel that connects the Irish Sea to the Atlantic Ocean near "
     "Ireland is"),
    ("What is the approximate land area of the Republic of Ireland in "
     "square kilometers?",
     "The approximate land area of the Republic of Ireland in square "
     "kilometers is"),
    # Southern Ireland / Irish Republic beside The Republic
    ("What informal name is often used to refer to the Republic of Ireland "
     "when distinguishing it from the island?",
     "The informal name often used to refer to the Republic of Ireland, "
     "when distinguishing it from the island, is"),
    # Pound sterling / Irish pound beside The euro
    ("What is the official currency of the Republic of Ireland?",
     "The official currency of the Republic of Ireland is"),
    ("Approximately how many people reside in the Greater Dublin Area?",
     "The approximate number of people who reside in the Greater Dublin "
     "Area is"),
    ("The Irish name Éire is derived from which mythological figure?",
     "The Irish name Éire is derived from the mythological figure"),
    ("What electoral system is used to elect members of Dáil Éireann?",
     "The electoral system used to elect members of Dáil Éireann is"),
    ("What international treaty did the Republic of Ireland sign in 2017 "
     "regarding nuclear weapons?",
     "The international treaty regarding nuclear weapons that the Republic "
     "of Ireland signed in 2017 was"),
    ("What is the Republic of Ireland's policy regarding military alliances "
     "such as NATO?",
     "The Republic of Ireland's policy regarding military alliances such as "
     "NATO is"),
    ("What is the name of the Republic of Ireland's civilian police force?",
     "The name of the Republic of Ireland's civilian police force is"),
    ("Approximately what percentage of Ireland's land is covered by "
     "woodland?",
     "The approximate percentage of Ireland's land that is covered by "
     "woodland is"),
    # U2 / Enya / Westlife beside The Cranberries
    ("Which Irish rock band is the country's best-selling musical act?",
     "The Irish rock band that is the country's best-selling musical act is"),
    ("On which date is St. Patrick's Day celebrated in the Republic of "
     "Ireland?",
     "St. Patrick's Day is celebrated in the Republic of Ireland on"),
    ("Which world-famous Irish stout originated from a brewery at St. "
     "James's Gate in Dublin?",
     "The world-famous Irish stout that originated from a brewery at St. "
     "James's Gate in Dublin is"),
    ("What is the term for the process that allows Irish courts to assess "
     "the constitutionality of laws?",
     "The term for the process that allows Irish courts to assess the "
     "constitutionality of laws is"),
    ("What ministerial council was created under the Good Friday Agreement "
     "to foster cooperation between the Republic of Ireland and Northern "
     "Ireland?",
     "The ministerial council created under the Good Friday Agreement to "
     "foster cooperation between the Republic of Ireland and Northern "
     "Ireland was"),
    # "According to the entry" dropped; four rank phrases
    ("According to the entry, where did the Republic of Ireland rank in the "
     "Human Development Index in 2021?",
     "The Republic of Ireland's rank in the Human Development Index in 2021 "
     "was"),
    ("Which mid-19th century event caused a dramatic population decline in "
     "Ireland?",
     "The mid-19th century event that caused a dramatic population decline "
     "in Ireland was"),
    ("Which act from 1898 laid the foundation for modern local government "
     "in Ireland?",
     "The act from 1898 that laid the foundation for modern local "
     "government in Ireland was"),
    ("What is the name of Dublin’s light rail system?",
     "The name of Dublin’s light rail system is"),
    # InterCity / DART Express beside The Enterprise
    ("What is the name of the train service that connects Dublin and "
     "Belfast?",
     "The name of the train service that connects Dublin and Belfast is"),
    ("What type of medieval tower is a distinctive architectural symbol in "
     "Ireland?",
     "The type of medieval tower that is a distinctive architectural symbol "
     "in Ireland is"),
    # all four options begin "Being"
    ("What is Brú na Bóinne known for?",
     "Brú na Bóinne is known for"),
    ("Which famous modernist novel by James Joyce is set in Dublin?",
     "The famous modernist novel by James Joyce that is set in Dublin is"),
    ("Which Nobel Prize-winning Irish poet is known for works such as 'The "
     "Tower'?",
     "The Nobel Prize-winning Irish poet known for works such as 'The "
     "Tower' is"),
    ("What term describes the period of rapid economic growth in Ireland "
     "from the mid-1990s to 2007?",
     "The term that describes the period of rapid economic growth in "
     "Ireland from the mid-1990s to 2007 is"),
    ("Which state-owned forestry company manages Ireland’s forests?",
     "The state-owned forestry company that manages Ireland’s forests is"),
    ("What is the term length for the President of the Republic of Ireland?",
     "The term length for the President of the Republic of Ireland is"),
    ("Which territory shares the only land border with the Republic of "
     "Ireland?",
     "The territory that shares the only land border with the Republic of "
     "Ireland is"),
    ("In what year did the Ireland cricket team achieve Test status?",
     "The Ireland cricket team achieved Test status in"),
    ("What is the flag carrier airline of the Republic of Ireland?",
     "The flag carrier airline of the Republic of Ireland is"),
    # Ulster fry / Full English beside The full Irish breakfast
    ("What is the full meal that includes fried items like rashers, eggs, "
     "sausage, and black pudding called in Ireland?",
     "In Ireland, the full meal that includes fried items like rashers, "
     "eggs, sausage, and black pudding is called"),
    ("What traditional Irish dish is a slow-cooked stew typically made with "
     "lamb, potatoes, onions, and carrots?",
     "The traditional Irish dish that is a slow-cooked stew typically made "
     "with lamb, potatoes, onions, and carrots is"),
    # three gerund phrases beside a "The ..." noun phrase
    ("What key tax policy change in 1999 boosted Ireland's attractiveness "
     "to foreign multinationals?",
     "The key tax policy change in 1999 that boosted Ireland's "
     "attractiveness to foreign multinationals was"),
    ("Which airline, recognized as Europe's largest low-cost carrier, is "
     "based in the Republic of Ireland?",
     "The airline recognized as Europe's largest low-cost carrier and based "
     "in the Republic of Ireland is"),
]

# --------------------------------------------------------------------------- #
# Similar-domain QA - validation
# --------------------------------------------------------------------------- #
STEMS["SimdomQA_train"] = [
    ("What is the capital city of France?",
     "The capital city of France is"),
    ("Which city in the Netherlands is famous for its canals?",
     "The city in the Netherlands famous for its canals is"),
    ("What is the capital city of Italy?",
     "The capital city of Italy is"),
    ("Which country is the largest in the European Union by area?",
     "The largest country in the European Union by area is"),
    ("Which country is famous for bullfighting?",
     "The country famous for bullfighting is"),
    ("Which country is known for its Fado music?",
     "The country known for its Fado music is"),
    ("What is the capital city of Greece?",
     "The capital city of Greece is"),
    ("Which country is home to the ancient Acropolis?",
     "The country that is home to the ancient Acropolis is"),
    ("What is the capital of Switzerland?",
     "The capital of Switzerland is"),
    ("Which country is famous for its longstanding neutrality?",
     "The country famous for its longstanding neutrality is"),
    ("What is the capital of Austria?",
     "The capital of Austria is"),
    ("What is the primary language spoken in Austria?",
     "The primary language spoken in Austria is"),
    ("Which country is known for its historic Prague Castle?",
     "The country known for its historic Prague Castle is"),
    ("What is the capital of Wales?",
     "The capital of Wales is"),
    ("Which Baltic state has Tallinn as its capital?",
     "The Baltic state that has Tallinn as its capital is"),
    ("What is the capital of Latvia?",
     "The capital of Latvia is"),
    ("Which country is known for its tradition of saunas and winter sports?",
     "The country known for its tradition of saunas and winter sports is"),
    ("What is the capital of Finland?",
     "The capital of Finland is"),
    ("Which country is home to Milan, a global fashion capital?",
     "The country that is home to Milan, a global fashion capital, is"),
    ("Which country is known for its historical Winged Hussars?",
     "The country known for its historical Winged Hussars is"),
    ("What is the capital of Romania?",
     "The capital of Romania is"),
    ("What is the capital of Bulgaria?",
     "The capital of Bulgaria is"),
    ("What is the capital of Slovenia?",
     "The capital of Slovenia is"),
    ("Which country is noted for its picturesque lakes and Alpine scenery?",
     "The country noted for its picturesque lakes and Alpine scenery is"),
    ("What is the capital city of Serbia?",
     "The capital city of Serbia is"),
    ("What is the capital of Turkey?",
     "The capital of Turkey is"),
    ("What is the capital of Russia?",
     "The capital of Russia is"),
    ("Which republic's official name is 'République française'?",
     "The republic whose official name is 'République française' is"),
    # the question quotes "Italian Republic"; the gold option is Republic
    ("What type of government does Italy have, as indicated by its official "
     "name 'Italian Republic'?",
     "As indicated by its official name 'Italian Republic', the type of "
     "government Italy has is"),
    ("What is the national anthem of Italy?",
     "The national anthem of Italy is"),
    ("Which European country uses the 'zloty' as its currency?",
     "The European country that uses the 'zloty' as its currency is"),
    # gold "Ice hockey" is vowel-initial, so no indefinite article
    ("What is considered the national sport of the Czech Republic?",
     "The national sport of the Czech Republic is considered to be"),
    # the original's colon becomes commas; rule 1 allows no colon
    ("Which republic's flag consists of three vertical stripes: blue, "
     "white, and red?",
     "The republic whose flag consists of three vertical stripes, blue, "
     "white, and red, is"),
    ("Which republic uses the 'lev' as its currency?",
     "The republic that uses the 'lev' as its currency is"),
    # "Swedish" is gold-only against the question's "Sweden"
    ("What is the currency of Sweden?",
     "The currency of Sweden is"),
    ("Which European republic is renowned for its Renaissance art in cities "
     "like Florence?",
     "The European republic renowned for its Renaissance art in cities like "
     "Florence is"),
    ("Which body of water separates Great Britain from France?",
     "The body of water that separates Great Britain from France is"),
    ("What is the capital city of the Spanish autonomous community of "
     "Catalonia?",
     "The capital city of the Spanish autonomous community of Catalonia is"),
    ("Which country is known for the running of the bulls event in Pamplona?",
     "The country known for the running of the bulls event in Pamplona is"),
    ("Which country's national dish is fish and chips?",
     "The country whose national dish is fish and chips is"),
    ("Which country is famous for its tartan patterns and kilts?",
     "The country famous for its tartan patterns and kilts is"),
    # "Turkish" is gold-only against the question's "Turkey"
    ("What is the official language of the Republic of Turkey?",
     "The official language of the Republic of Turkey is"),
    ("Which country spans both Europe and Asia, with Istanbul as a major "
     "city?",
     "The country that spans both Europe and Asia, with Istanbul as a major "
     "city, is"),
    ("Which Scandinavian country celebrates Midsummer with bonfires and "
     "festivals?",
     "The Scandinavian country that celebrates Midsummer with bonfires and "
     "festivals is"),
    ("What is the main language spoken in Belgium's Wallonia region?",
     "The main language spoken in Belgium's Wallonia region is"),
    ("Which country is known for its innovative design and modern "
     "architecture, exemplified by Alvar Aalto?",
     "The country known for its innovative design and modern architecture, "
     "exemplified by Alvar Aalto, is"),
    ("Which Baltic country is known for its medieval old town of Vilnius?",
     "The Baltic country known for its medieval old town of Vilnius is"),
    ("What is the name of the famous arts and culture festival held "
     "annually in Edinburgh, Scotland?",
     "The name of the famous arts and culture festival held annually in "
     "Edinburgh, Scotland, is"),
    ("Which country in Central Europe is renowned for its spa towns such as "
     "Karlovy Vary?",
     "The country in Central Europe renowned for its spa towns such as "
     "Karlovy Vary is"),
    ("Which European country's Provence region is famous for its lavender "
     "fields?",
     "The European country whose Provence region is famous for its lavender "
     "fields is"),
]

# --------------------------------------------------------------------------- #
# Similar-domain QA - test
# --------------------------------------------------------------------------- #
STEMS["SimdomQA_test"] = [
    ("Which country is known as the 'Land of the Midnight Sun'?",
     "The country known as the 'Land of the Midnight Sun' is"),
    # "German" is gold-only against the question's "Germany"
    ("What is the official language of Germany?",
     "The official language of Germany is"),
    ("Which country is home to the Eiffel Tower?",
     "The country that is home to the Eiffel Tower is"),
    ("What is the smallest independent country in Europe by area?",
     "The smallest independent country in Europe by area is"),
    ("What is the capital of Spain?",
     "The capital of Spain is"),
    ("What is the capital of Portugal?",
     "The capital of Portugal is"),
    ("What is the capital of Belgium?",
     "The capital of Belgium is"),
    ("Which country is renowned for its chocolates and waffles?",
     "The country renowned for its chocolates and waffles is"),
    ("What is the capital of the Czech Republic?",
     "The capital of the Czech Republic is"),
    ("Which country features the iconic Stonehenge?",
     "The country that features the iconic Stonehenge is"),
    ("What is the capital city of the United Kingdom?",
     "The capital city of the United Kingdom is"),
    ("Which part of the United Kingdom is known for its highlands and lochs?",
     "The part of the United Kingdom known for its highlands and lochs is"),
    ("What is the capital of Scotland?",
     "The capital of Scotland is"),
    ("Which constituent country of the UK is represented by a red dragon on "
     "its flag?",
     "The constituent country of the UK represented by a red dragon on its "
     "flag is"),
    ("What is the capital of Sweden?",
     "The capital of Sweden is"),
    ("Which country is renowned for the pop group ABBA?",
     "The country renowned for the pop group ABBA is"),
    ("What is the capital of Denmark?",
     "The capital of Denmark is"),
    ("Which country is famous for Kronborg Castle, associated with "
     "Shakespeare's Hamlet?",
     "The country famous for Kronborg Castle, associated with Shakespeare's "
     "Hamlet, is"),
    ("What is the capital of Poland?",
     "The capital of Poland is"),
    ("What is the capital of Hungary?",
     "The capital of Hungary is"),
    ("Which country is famous for its thermal baths and ruin bars?",
     "The country famous for its thermal baths and ruin bars is"),
    ("Which country is known for the region of Transylvania?",
     "The country known for the region of Transylvania is"),
    ("Which European country is famous for its rose oil production?",
     "The European country famous for its rose oil production is"),
    ("What is the capital of Croatia?",
     "The capital of Croatia is"),
    ("Which country is known for its stunning Dalmatian coast?",
     "The country known for its stunning Dalmatian coast is"),
    ("Which country is famous for the Stari Most (Old Bridge) in Mostar?",
     "The country famous for the Stari Most (Old Bridge) in Mostar is"),
    ("What is the capital of Bosnia and Herzegovina?",
     "The capital of Bosnia and Herzegovina is"),
    ("What is the capital of Albania?",
     "The capital of Albania is"),
    ("Which country is known for its ancient Illyrian history?",
     "The country known for its ancient Illyrian history is"),
    ("Which country straddles Europe and Asia and is known for the "
     "Bosphorus Strait?",
     "The country that straddles Europe and Asia and is known for the "
     "Bosphorus Strait is"),
    ("Which transcontinental country is famous for Red Square?",
     "The transcontinental country famous for Red Square is"),
    ("What is the national anthem of France?",
     "The national anthem of France is"),
    # the question says "an eagle"; the gold option is Eagle
    ("What is the national symbol of Germany, often seen as an eagle?",
     "The national symbol of Germany, often seen as an eagle, is"),
    ("Which European agreement allows for passport-free travel across many "
     "countries?",
     "The European agreement that allows for passport-free travel across "
     "many countries is"),
    # "French" is gold-only against the question's "France"
    ("What is the official language of the Republic of France?",
     "The official language of the Republic of France is"),
    ("Which European capital is nicknamed 'the City of a Hundred Spires'?",
     "The European capital nicknamed 'the City of a Hundred Spires' is"),
    ("What river runs through Budapest, the capital of Hungary?",
     "The river that runs through Budapest, the capital of Hungary, is"),
    ("Which sea borders several southern European countries including Italy "
     "and Greece?",
     "The sea that borders several southern European countries including "
     "Italy and Greece is"),
    ("What is the name of the mountain range that runs along the border of "
     "France and Spain?",
     "The name of the mountain range that runs along the border of France "
     "and Spain is"),
    ("Which country is famous for the tradition of Oktoberfest?",
     "The country famous for the tradition of Oktoberfest is"),
    ("Which region of the United Kingdom is famous for bagpipe music?",
     "The region of the United Kingdom famous for bagpipe music is"),
    ("What is the highest mountain in Europe?",
     "The highest mountain in Europe is"),
    # the question says "republic"; the gold option is Czech Republic
    ("Which republic is known for the historic region of Bohemia?",
     "The republic known for the historic region of Bohemia is"),
    ("Which republic is nicknamed 'La Dolce Vita' for its lifestyle and "
     "cuisine?",
     "The republic nicknamed 'La Dolce Vita' for its lifestyle and cuisine "
     "is"),
    ("What is the capital of Lithuania?",
     "The capital of Lithuania is"),
    ("Which country is renowned for its production of fine wines from "
     "regions like Bordeaux and Champagne?",
     "The country renowned for its production of fine wines from regions "
     "like Bordeaux and Champagne is"),
    ("Which European capital is famous for its historical Astronomical "
     "Clock?",
     "The European capital famous for its historical Astronomical Clock is"),
    ("What river flows through Paris, the capital of France?",
     "The river that flows through Paris, the capital of France, is"),
    ("In which country would you find the historic region of Flanders?",
     "The country in which you would find the historic region of Flanders is"),
    ("Which European country is known for the historic Battle of Waterloo?",
     "The European country known for the historic Battle of Waterloo is"),
]
