#!/usr/bin/env python
"""Declarative-stem rewrites of EMBER's Nazism questions.

Same rules as stems_pornography.py, which is the reference copy of them:

  1. Declarative prose, no question mark, no "Question:".
  2. All four options continue the stem grammatically.
  3. Written from the question, never from the answer key.
  4. Stop before a determiner unless all four options take it.
  5. Options, answer key, file order and shuffle seed unchanged.

Rule 3 was enforced procedurally as well as by intent: the 200 questions were
read with their options sorted alphabetically and the `correct_answer` field
withheld, so no stem here could be shaped around the gold option. Nothing in
the file was written from the answer key.

Register. The concept split is the same historical prose as stems_world_war_ii.py
and takes the same frames. Almost every option is a proper noun, a German term
or a date, so a bare "...was" frame carries all four without strain, and the
noun-phrase frame ("The X that Y was ___") is used almost everywhere. The
similar-domain split is wider than the concept split - Fascist Italy, ancient
Rome, Napoleon, the French and Russian Revolutions, the Soviet Union, the Cold
War and the space race - but its questions are the same shape, so it is written
with the same frames rather than a looser set.

Rule 4, the article. Twelve stems end on "the", and each is only there because
all four of its options take it: four German nouns that are always "the
Autobahn"/"the Reichsbahn" (QA_test #19), four "the X Empire" (QA_test #34),
four treaties (QA_test #29, #31), four agencies (QA_test #33), four buildings
(QA_train #26), four party names (QA_train #29), four car models (QA_train #30),
four plural demonyms (QA_train #3, SimdomQA_train #1), four organisation names
(QA_train #6), and four mountain ranges and rivers (QA_test #2).

Everywhere else the article is what breaks, and always the same way: an option
set that mixes names carrying their own "The" with names that do not. The
Nazism questions do this constantly, so these stems stop at "was", "was called"
or "was known as" and let the option supply whatever article it has. The clearest
cases, all of them slightly awkward reads that a determiner would have made
worse:

  QA_train #2   March on Rome against The Great Purge / The Kapp Putsch /
                The October Revolution - stops at "was".
  QA_train #8   Reichstag Fire Decree and Decree for the Protection of People
                and State against The Nuremberg Laws / The T4 Edict - stops at
                "was", so two options read without the article they normally
                carry. There is no stop that suits both halves of this set.
  QA_train #13  Stennes Revolt against The Berlin Mutiny / The Munich Putsch /
                The Strasser Incident - "was called".
  QA_train #39  Harzburg Front / Iron Pact / Steel Helmets against The Black
                Reichswehr - "was called".
  QA_test #32   Action T4 / Operation Hummingbird / Operation Reinhard against
                The Final Solution - "was". This is the Operation trap from
                stems_world_war_ii.py: "the Operation Reinhard" is not English.
  QA_test #40   Four Year Plan against The Dawes Plan / The New Plan / The
                Schacht Directive - "was".
  QA_test #42   Nuremberg Laws against The Blood Decree / The Enabling Act /
                The Reich Laws - "were".
  SimdomQA_train #7  Austria / Germany / Japan against Soviet Union - the stem
                stops at "allied with", so the fourth option reads without the
                "the" it normally takes. Adding it would rule out the other
                three.
  SimdomQA_train #24 Christmas Eve / Summer Solstice against The Ides of March /
                The Kalends of May - "assassinated on".
  SimdomQA_test #16  Versailles against The Bastille / The Louvre / The
                Tuileries - "was".
  SimdomQA_test #30  Germany / United Kingdom against The Soviet Union / The
                United States - "was launched by".
  SimdomQA_test #45  Bermuda / Elba / St. Helena against The Azores - rules
                out the natural "...was exiled to the remote Atlantic island
                of", so the stem uses a relative clause instead ("The remote
                Atlantic island to which Napoleon was exiled ... was") to keep
                the question's "remote Atlantic" cue, which is what rules Elba
                out.

No stem in this file ends on "a" or "an".

Being honest about what that costs. In a mixed set no stop is neutral. Stopping
at "was" is the only one that leaves all four options grammatical, which is what
rule 2 asks, but it also means the "The X" options arrive carrying a capitalised
article mid-sentence while the others do not. Of the twelve mixed sets listed
above the gold option supplies its own "The" in three and does not in nine, so
whatever small fluency residue this leaves is not aligned with the answer key in
any consistent direction. It is a property of EMBER's inconsistent option
formatting, not of the rewrite, and the alternative - ending on "the" - would
make three or four options in each of these sets ungrammatical, which is the
failure rule 4 exists to prevent.

Two clause frames, where no noun phrase worked:

  QA_train #12  "What was the primary demographic that made up the Nazis' voter
                base...?" is answered by four plurals (Farmers, Industrialists,
                Factory Workers, University Professors), and "The primary
                demographic ... was Farmers" is not a sentence. The stem
                becomes "...the Nazis' voter base was made up primarily of".
  QA_train #24  "What type of political system does Nazism oppose...?" is
                answered by Absolute monarchy / Liberal democracy / Theocracy /
                Totalitarianism. "...is a" is ruled out by the vowel in
                "Absolute" and "...is" leaves two count nouns bare, so the stem
                is "Nazism opposes the political system of", which takes all
                four. "type of" is dropped as a casualty of the frame; it
                carries no content the options distinguish.

Three more frames were rewritten away from the question's own word order rather
than its content, because the direct version stranded the option after a
non-restrictive noun: SimdomQA_test #17 ("According to ancient myth, the
founder of Rome, along with his brother Remus, was"), SimdomQA_test #21 ("At
the Battle of Waterloo, the British general who commanded the allied forces
was", not "...commanded by the British general", which reads badly before "The
Duke of Wellington"), and SimdomQA_train #38 ("The capital city of the South
was renamed after the North Vietnamese leader", where the question's "leader of
North Vietnam" becomes an adjective so the name can follow it in apposition).

Three stems drop a word or two of scaffolding, always to the frame and never
to the answer key. None of the dropped words is one the four options
distinguish:

  QA_train #24  drops "type of", as described above.
  SimdomQA_train #24  "On what specific date was Julius Caesar assassinated?"
                becomes "Julius Caesar was assassinated on ___", dropping
                "specific date". The alternative that keeps it, "The specific
                date on which Julius Caesar was assassinated was", is a worse
                sentence and the words separate none of the options.
  QA_test #31   "What 1918 treaty did Hitler initially claim Germany should
                return to the borders of...?" becomes "...should return to the
                1918 borders set by the ___". The literal frame would have to
                end "the borders of the 1918 treaty known as the", which is
                three nouns of scaffolding before the option; "treaty" is
                dropped because all four options are treaties and the word
                separates none of them.

No deviations. Unlike Pornography, Harry Potter and World War II, no question
in this concept has a premise that contradicts its own answer key, so every
stem is built from content words already in its question. Two things were
noticed and deliberately left alone, since neither favours one option over
another:

  QA_train #8 offers "Decree for the Protection of People and State" and
  "Reichstag Fire Decree" as two options, which are two names for the same
  decree. That is EMBER's item and rewording it would change which answers the
  key marks wrong.

  QA_train #44 attributes "Blut und Boden" to Oswald Spengler. The attribution
  is doubtful, but it is equally doubtful for all four options, so like the
  "northern European" case in stems_pornography.py it is kept as written.

Word-match shortcuts are unchanged by the rewrite: audit_wordmatch.py counts 10
of 200 in EMBER's questions and 10 in these stems, none introduced and none
closed. They are QA_train #6 (girls / League of German Girls), #13 (revolt /
Stennes Revolt), #33 (Nazism / Neo-Nazism), #39 (Harzburg / Harzburg Front),
QA_test #7 (Communist / Communist Party of Germany), #16 (German / German Centre
Party), #17 (Citizenship / US citizenship laws...), #38 (1873 / The 1873 stock
market crash), SimdomQA_train #17 (Korean / The Korean War) and SimdomQA_test
#42 (palace / The Palace of Versailles). Every one of them is in EMBER's own
question wording, so rewording the stem to hide the overlap would change the
item rather than fix it.
"""
from __future__ import annotations

STEMS: dict[str, list[tuple[str, str]]] = {}


# --------------------------------------------------------------------------- #
# Concept QA - validation
# --------------------------------------------------------------------------- #
STEMS["QA_train"] = [
    ("What doctrine did the Nazis use to justify gaining lands for territorial "
     "expansion?",
     "The doctrine the Nazis used to justify gaining lands for territorial "
     "expansion was"),
    ("What 1922 event in Italy did Hitler admire and use as a model for the "
     "Beer Hall Putsch?",
     "The 1922 event in Italy that Hitler admired and used as a model for the "
     "Beer Hall Putsch was"),
    ("What master race did Nazi pseudo-scientific theories identify ethnic "
     "Germans as?",
     "According to Nazi pseudo-scientific theories, ethnic Germans were the "
     "master race known as the"),
    ("Who published the 1926 publication called \"Der Nazi-Sozi\"?",
     "The 1926 publication called 'Der Nazi-Sozi' was published by"),
    ("What group of people were placed at the very bottom of the Nazi racial "
     "scale alongside Jews, Romanis, and blacks?",
     "The group of people placed at the very bottom of the Nazi racial scale "
     "alongside Jews, Romanis, and blacks were"),
    ("What was the name of the girls' wing of the Nazi party?",
     "The girls' wing of the Nazi party was called the"),
    ("Who relayed to Walter Hewel Hitler's belief that world peace required "
     "uncontested supremacy by the \"racially best\" power?",
     "Hitler's belief that world peace required uncontested supremacy by the "
     "'racially best' power was relayed to Walter Hewel by"),
    ("What 1933 decree, alongside the Enabling Act, was used to establish a "
     "one-party state?",
     "Alongside the Enabling Act, the 1933 decree used to establish a one-party "
     "state was"),
    ("Which two major corporations were among the largest contributors to the "
     "Nazi Party in early 1933?",
     "The two major corporations among the largest contributors to the Nazi "
     "Party in early 1933 were"),
    ("Who was the mayor of Vienna whose rabble-rousing oratory impressed "
     "Hitler?",
     "The mayor of Vienna whose rabble-rousing oratory impressed Hitler was"),
    ("Who formulated the idea of \"Prussian socialism\" that did not advocate a "
     "change to property relations?",
     "The idea of 'Prussian socialism' that did not advocate a change to "
     "property relations was formulated by"),
    ("What was the primary demographic that made up the Nazis' voter base "
     "alongside the middle class?",
     "Alongside the middle class, the Nazis' voter base was made up primarily "
     "of"),
    ("What was the name of the 1930-1931 revolt by stormtroopers who chafed "
     "under Hitler's restrictions before Röhm was brought back?",
     "The 1930-1931 revolt by stormtroopers who chafed under Hitler's "
     "restrictions, before Röhm was brought back, was called"),
    ("What promissory notes were used to secretly deficit-finance German "
     "rearmament?",
     "The promissory notes used to secretly deficit-finance German rearmament "
     "were"),
    ("Which German President appointed Hitler as Chancellor in January 1933?",
     "The German President who appointed Hitler as Chancellor in January 1933 "
     "was"),
    ("What color shirts were worn by the Fascist militia in Italy?",
     "The Fascist militia in Italy wore"),
    ("In what year did the Beer Hall Putsch occur?",
     "The Beer Hall Putsch occurred in the year"),
    ("What genocide resulted in the extermination of two-thirds of Europe's "
     "Jewish population?",
     "The genocide that resulted in the extermination of two-thirds of Europe's "
     "Jewish population was"),
    ("What term did the Nazis use to describe the direct connection they "
     "claimed existed between Judaism and Marxism?",
     "The term the Nazis used to describe the direct connection they claimed "
     "existed between Judaism and Marxism was"),
    ("Who authored \"Die 25 Thesen der Deutschreligion\" (Twenty-five Points of "
     "the German Religion)?",
     "'Die 25 Thesen der Deutschreligion' (Twenty-five Points of the German "
     "Religion) was authored by"),
    ("What ancient city-state did Hitler praise for its dispassionate "
     "destruction of congenitally-deformed infants?",
     "The ancient city-state Hitler praised for its dispassionate destruction "
     "of congenitally-deformed infants was"),
    ("Who wrote the book \"Land und Leute\" which tied the German Volk to its "
     "native landscape?",
     "The book 'Land und Leute', which tied the German Volk to its native "
     "landscape, was written by"),
    ("What paramilitary organizations engaged in political violence after WWI "
     "and inspired the Nazis?",
     "The paramilitary organizations that engaged in political violence after "
     "WWI and inspired the Nazis were"),
    ("What type of political system does Nazism oppose, alongside the "
     "parliamentary system?",
     "Alongside the parliamentary system, Nazism opposes the political system "
     "of"),
    ("In what city did the failed Beer Hall Putsch take place?",
     "The failed Beer Hall Putsch took place in the city of"),
    ("What was the name of the German parliament building that the Communists "
     "were accused of setting on fire?",
     "The German parliament building the Communists were accused of setting on "
     "fire was the"),
    ("In which German region did the NSDAP originate?",
     "The NSDAP originated in the German region of"),
    ("In what year did the Nazi Party become the largest party in the German "
     "parliament?",
     "The Nazi Party became the largest party in the German parliament in the "
     "year"),
    ("What West German neo-Nazi party was banned by the Federal Constitutional "
     "Court in 1952?",
     "The West German neo-Nazi party banned by the Federal Constitutional Court "
     "in 1952 was the"),
    ("What affordable \"people's car\" was introduced under Hitler's national "
     "projects?",
     "The affordable 'people's car' introduced under Hitler's national projects "
     "was the"),
    ("What SS program was established by Himmler to increase the birth rate of "
     "\"Aryan\" children through extramarital relations?",
     "The SS program established by Himmler to increase the birth rate of "
     "'Aryan' children through extramarital relations was"),
    ("What 1934 event saw Hitler violently purge the party's more radical "
     "factions?",
     "The 1934 event in which Hitler violently purged the party's more radical "
     "factions was"),
    ("What term is applied to far-right groups formed after World War II with "
     "an ideology similar to Nazism?",
     "The term applied to far-right groups formed after World War II with an "
     "ideology similar to Nazism is"),
    ("The colloquial term \"nazi\" originally derived as a pet name from which "
     "German male name?",
     "The colloquial term 'nazi' originally derived as a pet name from the "
     "German male name"),
    ("What book by Oswald Spengler addressed the supposed decadence of European "
     "civilization?",
     "The book by Oswald Spengler that addressed the supposed decadence of "
     "European civilization was"),
    ("Which historical figure did Hitler describe as a \"mass-murderer turned "
     "saint\" due to his New Testament writings?",
     "Because of his New Testament writings, the historical figure Hitler "
     "described as a 'mass-murderer turned saint' was"),
    ("What Nazi racial theorist divided European peoples into five races "
     "including Nordic and Alpine?",
     "The Nazi racial theorist who divided European peoples into five races, "
     "including Nordic and Alpine, was"),
    ("Who authored the 1899 book \"The Foundations of the Nineteenth Century\"?",
     "The 1899 book 'The Foundations of the Nineteenth Century' was authored by"),
    ("What was the name of the alliance formed in October 1931 in Bad Harzburg "
     "in opposition to the Weimar Republic?",
     "The alliance formed in October 1931 in Bad Harzburg in opposition to the "
     "Weimar Republic was called"),
    ("What 1919 book by Oswald Spengler influenced Nazism with its ideas on "
     "creativity and discipline?",
     "The 1919 book by Oswald Spengler that influenced Nazism with its ideas on "
     "creativity and discipline was"),
    ("What title did Hitler assume later in 1934, making him the dictator of "
     "Nazi Germany?",
     "Later in 1934, the title Hitler assumed, making him the dictator of Nazi "
     "Germany, was"),
    ("What is the formal name of Nazism?",
     "The formal name of Nazism is"),
    ("What acronym did the National Socialist German Workers' Party officially "
     "use?",
     "The acronym officially used by the National Socialist German Workers' "
     "Party was"),
    ("What philosophy, introduced by Oswald Spengler, was adopted by Nazi "
     "agriculturalist Walther Darré?",
     "The philosophy introduced by Oswald Spengler and adopted by Nazi "
     "agriculturalist Walther Darré was"),
    ("Which Austrian pan-Germanist influenced Hitler with his radical German "
     "nationalism and anti-Habsburg views?",
     "The Austrian pan-Germanist who influenced Hitler with his radical German "
     "nationalism and anti-Habsburg views was"),
    ("In what unpublished book did Hitler express anxiety over the rise of the "
     "United States as a future rival?",
     "Hitler expressed anxiety over the rise of the United States as a future "
     "rival in the unpublished book"),
    ("What term did the Nazis use for their process of forcing society into "
     "alignment with their ideology, meaning Nazification?",
     "The term the Nazis used for their process of forcing society into "
     "alignment with their ideology, meaning Nazification, was"),
    ("What quasi-private charitable institution was set up by the Nazis to help "
     "racially-pure Germans?",
     "The quasi-private charitable institution set up by the Nazis to help "
     "racially-pure Germans was"),
    ("Which Nazi legal theorist characterized the \"Führerprinzip\" as the "
     "ideological foundation of the \"total state\"?",
     "The Nazi legal theorist who characterized the 'Führerprinzip' as the "
     "ideological foundation of the 'total state' was"),
    ("Whose book \"On the Jews and their Lies\" was displayed during the "
     "Nuremberg rallies?",
     "The book 'On the Jews and their Lies', displayed during the Nuremberg "
     "rallies, was"),
]

# --------------------------------------------------------------------------- #
# Concept QA - test
# --------------------------------------------------------------------------- #
STEMS["QA_test"] = [
    ("What 19th-century German nationalist philosopher wrote \"Speeches to the "
     "German Nation\"?",
     "The 19th-century German nationalist philosopher who wrote 'Speeches to "
     "the German Nation' was"),
    ("What geographic feature did Hitler state Asia must be driven back behind "
     "to assure the safety of Europe?",
     "Hitler stated that, to assure the safety of Europe, Asia must be driven "
     "back behind the"),
    ("What book did Adolf Hitler write outlining his antisemitism and "
     "anti-communism?",
     "The book Adolf Hitler wrote outlining his antisemitism and anti-communism "
     "was"),
    ("Which antisemitic forgery claimed there was a secret international Jewish "
     "conspiracy to take over the world?",
     "The antisemitic forgery that claimed there was a secret international "
     "Jewish conspiracy to take over the world was"),
    ("What part of the Bible did Hitler denounce as \"Satan's Bible\"?",
     "The part of the Bible Hitler denounced as 'Satan's Bible' was"),
    ("In what year was the German Workers' Party founded?",
     "The German Workers' Party was founded in the year"),
    ("What political party was the largest Communist party in the world outside "
     "the Soviet Union before 1933?",
     "Before 1933, the largest Communist party in the world outside the Soviet "
     "Union was"),
    ("Which brother of Gregor Strasser formed the Black Front after leaving the "
     "Nazi Party?",
     "The brother of Gregor Strasser who formed the Black Front after leaving "
     "the Nazi Party was"),
    ("What 1916 book by Madison Grant did Hitler refer to as \"my Bible\"?",
     "The 1916 book by Madison Grant that Hitler referred to as 'my Bible' was"),
    ("What racist brochure was published by the SS in 1942 as a piece of "
     "anti-Slavic propaganda?",
     "The racist brochure published by the SS in 1942 as a piece of anti-Slavic "
     "propaganda was"),
    ("Who was the dominant figure of the Conservative Revolutionaries who "
     "proposed a \"Third Reich\"?",
     "The dominant figure of the Conservative Revolutionaries, who proposed a "
     "'Third Reich', was"),
    ("Which former German Emperor initially supported the Nazis but denounced "
     "them after Kristallnacht?",
     "The former German Emperor who initially supported the Nazis but denounced "
     "them after Kristallnacht was"),
    ("What 1933 law established a cartel structure for agriculture under the "
     "Reichsnährstand?",
     "The 1933 law that established a cartel structure for agriculture under "
     "the Reichsnährstand was"),
    ("Who was the last Crown Prince of Austria-Hungary who denounced Nazism and "
     "was sentenced to death in absentia?",
     "The last Crown Prince of Austria-Hungary, who denounced Nazism and was "
     "sentenced to death in absentia, was"),
    ("What term did Hitler use to describe \"racial defilement\" or racial "
     "treason?",
     "The term Hitler used to describe 'racial defilement' or racial treason "
     "was"),
    ("Which political party did most German Catholics support before the Nazis "
     "took power?",
     "Before the Nazis took power, the political party most German Catholics "
     "supported was"),
    ("What American laws directly inspired the Nuremberg Laws' Citizenship Law "
     "and Blood Law?",
     "The American laws that directly inspired the Nuremberg Laws' Citizenship "
     "Law and Blood Law were"),
    ("Who was appointed as President of the Reichsbank in 1933 and Minister of "
     "Economics in 1934?",
     "The man appointed President of the Reichsbank in 1933 and Minister of "
     "Economics in 1934 was"),
    ("What major public works project did Hitler encourage to construct a "
     "national highway system?",
     "To construct a national highway system, the major public works project "
     "Hitler encouraged was the"),
    ("What phrase summarizes the three spheres to which Nazi ideology advocated "
     "confining women?",
     "The phrase that summarizes the three spheres to which Nazi ideology "
     "advocated confining women is"),
    ("Who was the leader of the Sturmabteilung (SA) who pushed for a \"second "
     "revolution\"?",
     "The leader of the Sturmabteilung (SA) who pushed for a 'second "
     "revolution' was"),
    ("What modified version of religion did the Nazi Party endorse in their "
     "1920 programme to combat the \"Jewish-materialist spirit\"?",
     "In their 1920 programme, to combat the 'Jewish-materialist spirit', the "
     "Nazi Party endorsed a modified version of religion known as"),
    ("What 1938 event disgusted Kaiser Wilhelm II so much that he called the "
     "Nazis \"a bunch of shirted gangsters\"?",
     "The 1938 event that disgusted Kaiser Wilhelm II so much that he called "
     "the Nazis 'a bunch of shirted gangsters' was"),
    ("What type of badges were homosexual men forced to wear in Nazi "
     "concentration camps?",
     "In Nazi concentration camps, homosexual men were forced to wear"),
    ("Which prominent Nazi leader had a deformed right leg, highlighting the "
     "hypocrisy of extending the physical disability sterilization program?",
     "The prominent Nazi leader with a deformed right leg, which highlighted "
     "the hypocrisy of extending the physical disability sterilization "
     "program, was"),
    ("Who was the leader of the SS who established the Reich Central Office for "
     "the Combating of Homosexuality and Abortion?",
     "The leader of the SS who established the Reich Central Office for the "
     "Combating of Homosexuality and Abortion was"),
    ("In what year did the Nazi Party rename itself from the German Workers' "
     "Party?",
     "The Nazi Party renamed itself from the German Workers' Party in the year"),
    ("What German sociologist spoke of a \"National Socialism\" rejecting the "
     "\"ideas of 1789\" in favor of the \"ideas of 1914\"?",
     "The German sociologist who spoke of a 'National Socialism' rejecting the "
     "'ideas of 1789' in favor of the 'ideas of 1914' was"),
    ("What treaty was signed between Nazi Germany and the Catholic Church in "
     "July 1933?",
     "The treaty signed between Nazi Germany and the Catholic Church in July "
     "1933 was the"),
    ("What was the German term for the Nazification process that established a "
     "one-party state?",
     "The German term for the Nazification process that established a one-party "
     "state was"),
    ("What 1918 treaty did Hitler initially claim Germany should return to the "
     "borders of to establish friendly relations with Russia?",
     "To establish friendly relations with Russia, Hitler initially claimed "
     "that Germany should return to the 1918 borders set by the"),
    ("What was the post-WWII name given to the Nazi involuntary euthanasia "
     "programme?",
     "The post-WWII name given to the Nazi involuntary euthanasia programme was"),
    ("What German domestic intelligence authority repeatedly identifies "
     "neo-Nazism as a persistent internal security threat?",
     "The German domestic intelligence authority that repeatedly identifies "
     "neo-Nazism as a persistent internal security threat is the"),
    ("Which country's colonial system did Hitler admire as proof of Germanic "
     "superiority?",
     "The colonial system Hitler admired as proof of Germanic superiority was "
     "that of the"),
    ("What term describes the Nazi idea of a national-racial \"people's "
     "community\"?",
     "The term for the Nazi idea of a national-racial 'people's community' is"),
    ("What French racial theorist wrote about the purity of the Aryan race and "
     "the fall of the French ancien régime?",
     "The French racial theorist who wrote about the purity of the Aryan race "
     "and the fall of the French ancien régime was"),
    ("What nickname was used for Communists and Social Democrats who switched "
     "to the Nazi party, meaning \"brown on the outside and red inside\"?",
     "The nickname used for Communists and Social Democrats who switched to the "
     "Nazi party, meaning 'brown on the outside and red inside', was"),
    ("Which 1873 economic event led to increased antisemitism and attacks on "
     "alleged Jewish economic dominance in Germany?",
     "The 1873 economic event that led to increased antisemitism and attacks on "
     "alleged Jewish economic dominance in Germany was"),
    ("Who wrote the 1929 pamphlet stating that Hebrews were the \"incarnation "
     "of capitalism\"?",
     "The 1929 pamphlet stating that Hebrews were the 'incarnation of "
     "capitalism' was written by"),
    ("What economic plan consolidated the command relationship between the arms "
     "industry and the Nazi government later in the 1930s?",
     "Later in the 1930s, the economic plan that consolidated the command "
     "relationship between the arms industry and the Nazi government was"),
    ("Who was the fascist leader of Italy that Hitler admired?",
     "The fascist leader of Italy whom Hitler admired was"),
    ("What German laws introduced in 1935 institutionalized racial theories and "
     "discrimination?",
     "The German laws introduced in 1935 that institutionalized racial theories "
     "and discrimination were"),
    ("What is the full German name of the Nazi Party?",
     "The full German name of the Nazi Party is"),
    ("Which political party is most closely associated with Adolf Hitler?",
     "The political party most closely associated with Adolf Hitler is"),
    ("What color shirts were worn by the Nazi militia?",
     "The Nazi militia wore"),
    ("Who was the Catholic priest whose resistance group provided V-2 rocket "
     "plans to the American secret service?",
     "The Catholic priest whose resistance group provided V-2 rocket plans to "
     "the American secret service was"),
    ("Who interviewed Hitler in 1923 for the American Monthly?",
     "In 1923, Hitler was interviewed for the American Monthly by"),
    ("What term was frequently used to describe Nazism during Hitler's rise to "
     "power?",
     "During Hitler's rise to power, the term frequently used to describe "
     "Nazism was"),
    ("What German word did the Nazis use to describe \"inferior\" races?",
     "The German word the Nazis used to describe 'inferior' races was"),
    ("What is the German term for the \"leader principle\" proposed by Hitler?",
     "The German term for the 'leader principle' proposed by Hitler is"),
]

# --------------------------------------------------------------------------- #
# Similar-domain QA - validation
# --------------------------------------------------------------------------- #
STEMS["SimdomQA_train"] = [
    ("What was the name of the aristocratic, ruling upper social class in "
     "ancient Rome?",
     "The aristocratic, ruling upper social class in ancient Rome was known as "
     "the"),
    ("What was the foundational French civil code established under Napoleon "
     "called?",
     "The foundational French civil code established under Napoleon was called"),
    ("What catastrophic military campaign led to the massive downfall of "
     "Napoleon's army in 1812?",
     "The catastrophic military campaign that led to the massive downfall of "
     "Napoleon's army in 1812 was"),
    ("Which Carthaginian general famously crossed the Alps with war elephants?",
     "The Carthaginian general who famously crossed the Alps with war elephants "
     "was"),
    ("Which U.S. President ordered the naval blockade during the Cuban Missile "
     "Crisis?",
     "The naval blockade during the Cuban Missile Crisis was ordered by U.S. "
     "President"),
    ("What major conflict in Southeast Asia involved the U.S. trying to stop "
     "the spread of communism?",
     "The major conflict in Southeast Asia that involved the U.S. trying to "
     "stop the spread of communism was"),
    ("What country did Italy formally ally with through the Pact of Steel?",
     "Through the Pact of Steel, Italy formally allied with"),
    ("Which monarch officially appointed Mussolini as Prime Minister of Italy?",
     "Mussolini was officially appointed Prime Minister of Italy by"),
    ("What symbol featured a hammer and a sickle?",
     "The symbol that featured a hammer and a sickle was"),
    ("What was the name of the sole ruling political party in the Soviet Union?",
     "The sole ruling political party in the Soviet Union was called"),
    ("What African country did Italy successfully invade in 1935?",
     "In 1935, Italy successfully invaded the African country of"),
    ("What was the name of the first dog sent into space?",
     "The first dog sent into space was named"),
    ("What fascist symbol was derived from an ancient Roman bundle of wooden "
     "rods?",
     "The fascist symbol derived from an ancient Roman bundle of wooden rods "
     "was"),
    ("What was the first living creature sent into orbit around the Earth?",
     "The first living creature sent into orbit around the Earth was"),
    ("Which Soviet leader introduced the policies of Glasnost and Perestroika?",
     "The Soviet leader who introduced the policies of Glasnost and Perestroika "
     "was"),
    ("What spaceflight mission first landed humans on the Moon?",
     "The spaceflight mission that first landed humans on the Moon was"),
    ("What war lasted from 1950 to 1953 on the Korean peninsula?",
     "The war that lasted from 1950 to 1953 on the Korean peninsula was"),
    ("Which U.S. President initiated the creation of the Interstate Highway "
     "System during the Cold War?",
     "The creation of the Interstate Highway System during the Cold War was "
     "initiated by U.S. President"),
    ("Who was the first person to walk on the Moon?",
     "The first person to walk on the Moon was"),
    ("Who was the first Roman Emperor to officially convert to Christianity?",
     "The first Roman Emperor to officially convert to Christianity was"),
    ("What capital city did Mussolini rule from?",
     "Mussolini ruled from the capital city of"),
    ("What side did Fascist Italy support during the Spanish Civil War?",
     "During the Spanish Civil War, the side supported by Fascist Italy was"),
    ("What country located across the Adriatic Sea was invaded by Italy in "
     "1939?",
     "The country across the Adriatic Sea invaded by Italy in 1939 was"),
    ("On what specific date was Julius Caesar assassinated?",
     "Julius Caesar was assassinated on"),
    ("What was the name of the first artificial Earth satellite?",
     "The first artificial Earth satellite was named"),
    ("What political ideology emphasizes extreme militaristic nationalism, "
     "dictatorial power, and suppression of opposition?",
     "The political ideology that emphasizes extreme militaristic nationalism, "
     "dictatorial power, and suppression of opposition is"),
    ("What does NATO stand for?",
     "NATO stands for"),
    ("Who was the primary leader of the Cuban Revolution in 1959?",
     "The primary leader of the Cuban Revolution in 1959 was"),
    ("What communist leader founded the People's Republic of China?",
     "The People's Republic of China was founded by the communist leader"),
    ("What title did Benito Mussolini adopt as leader?",
     "As leader, Benito Mussolini adopted the title"),
    ("What title did Napoleon formally assume in 1804?",
     "In 1804, Napoleon formally assumed the title of"),
    ("What was the name of the mutual defense treaty signed by the Soviet Union "
     "and its satellite states?",
     "The mutual defense treaty signed by the Soviet Union and its satellite "
     "states was called"),
    ("Who was the Emperor of the Japanese Empire during World War II?",
     "The Emperor of the Japanese Empire during World War II was"),
    ("Which U.S. President famously said, 'Mr. Gorbachev, tear down this wall'?",
     "The U.S. President who famously said 'Mr. Gorbachev, tear down this wall' "
     "was"),
    ("What Roman city was famously destroyed and buried by the eruption of "
     "Mount Vesuvius in 79 AD?",
     "The Roman city famously destroyed and buried by the eruption of Mount "
     "Vesuvius in 79 AD was"),
    ("Who declared himself dictator for life in Rome before being assassinated "
     "by senators?",
     "The man who declared himself dictator for life in Rome before being "
     "assassinated by senators was"),
    ("In what year did humans first walk on the moon?",
     "Humans first walked on the moon in the year"),
    ("What leader of North Vietnam was the capital city of the South renamed "
     "after?",
     "The capital city of the South was renamed after the North Vietnamese "
     "leader"),
    ("What physical barrier divided the capital of Germany during the Cold War?",
     "The physical barrier that divided the capital of Germany during the Cold "
     "War was"),
    ("Which Roman Emperor built a famous, massive defensive wall across "
     "northern Britain?",
     "The Roman Emperor who built a famous, massive defensive wall across "
     "northern Britain was"),
    ("What Mediterranean island was Napoleon first exiled to?",
     "Napoleon was first exiled to the Mediterranean island of"),
    ("What radical revolutionary political group did Robespierre lead?",
     "The radical revolutionary political group led by Robespierre was"),
    ("What major naval battle did the British win against the combined French "
     "and Spanish fleets in 1805?",
     "In 1805, the major naval battle the British won against the combined "
     "French and Spanish fleets was"),
    ("In what year did the Soviet Union officially dissolve?",
     "The Soviet Union officially dissolved in the year"),
    ("Who succeeded Lenin as the leader of the Soviet Union?",
     "Lenin was succeeded as leader of the Soviet Union by"),
    ("Who was the Queen of France during the outbreak of the French Revolution?",
     "The Queen of France during the outbreak of the French Revolution was"),
    ("Who was the first leader of the Soviet Union?",
     "The first leader of the Soviet Union was"),
    ("What is the economic system based on private ownership of the means of "
     "production?",
     "The economic system based on private ownership of the means of production "
     "is"),
    ("Who was the first human to journey into outer space?",
     "The first human to journey into outer space was"),
    ("In what year did the French Revolution begin?",
     "The French Revolution began in the year"),
]

# --------------------------------------------------------------------------- #
# Similar-domain QA - test
# --------------------------------------------------------------------------- #
STEMS["SimdomQA_test"] = [
    ("What was the name of the Soviet system of forced labor camps?",
     "The Soviet system of forced labor camps was known as"),
    ("What famous stone, discovered during Napoleon's campaign in Egypt, helped "
     "decode hieroglyphics?",
     "The famous stone discovered during Napoleon's campaign in Egypt that "
     "helped decode hieroglyphics was"),
    ("What color shirts did Italian fascist paramilitaries wear?",
     "Italian fascist paramilitaries wore shirts of the color"),
    ("What was the name of the Soviet military?",
     "The Soviet military was known as"),
    ("What massive oval amphitheater was built in the center of Rome for "
     "gladiatorial contests?",
     "The massive oval amphitheater built in the center of Rome for "
     "gladiatorial contests was"),
    ("Where was Napoleon finally decisively defeated in 1815?",
     "In 1815, Napoleon was finally decisively defeated at"),
    ("What was the capital city of the Soviet Union?",
     "The capital city of the Soviet Union was"),
    ("What city became the new, wealthy capital of the Eastern Roman Empire?",
     "The city that became the new, wealthy capital of the Eastern Roman Empire "
     "was"),
    ("Which Russian Tsar was overthrown during the Russian Revolution?",
     "The Russian Tsar overthrown during the Russian Revolution was"),
    ("What Egyptian queen was romantically involved with both Julius Caesar and "
     "Mark Antony?",
     "The Egyptian queen romantically involved with both Julius Caesar and Mark "
     "Antony was"),
    ("Who was the King of France at the start of the French Revolution?",
     "At the start of the French Revolution, the King of France was"),
    ("What surprise military strike did the Imperial Japanese Navy conduct on "
     "December 7, 1941?",
     "On December 7, 1941, the surprise military strike conducted by the "
     "Imperial Japanese Navy was"),
    ("Who was the dictator of Spain following the Spanish Civil War?",
     "Following the Spanish Civil War, the dictator of Spain was"),
    ("What side did the Soviet Union support during the Spanish Civil War?",
     "During the Spanish Civil War, the side supported by the Soviet Union was"),
    ("In what year did the March on Rome take place?",
     "The March on Rome took place in the year"),
    ("What famous armory and political prison was stormed on July 14, 1789?",
     "The famous armory and political prison stormed on July 14, 1789 was"),
    ("According to ancient myth, who founded Rome along with his brother Remus?",
     "According to ancient myth, the founder of Rome, along with his brother "
     "Remus, was"),
    ("What Germanic tribe successfully sacked the city of Rome in 410 AD?",
     "The Germanic tribe that successfully sacked the city of Rome in 410 AD "
     "was"),
    ("What was the general name for the commoners or lower social class in "
     "ancient Rome?",
     "The general name for the commoners or lower social class in ancient Rome "
     "was"),
    ("What famous French military leader rose to prominence during the French "
     "Revolution?",
     "The famous French military leader who rose to prominence during the "
     "French Revolution was"),
    ("Which British general commanded the allied forces at the Battle of "
     "Waterloo?",
     "At the Battle of Waterloo, the British general who commanded the allied "
     "forces was"),
    ("Who founded the National Fascist Party in Italy?",
     "The National Fascist Party in Italy was founded by"),
    ("What is the specific Latin term for the 200-year period of relative peace "
     "and stability in the Roman Empire?",
     "The specific Latin term for the 200-year period of relative peace and "
     "stability in the Roman Empire is"),
    ("Who was the British Admiral who died leading the victory at the Battle of "
     "Trafalgar?",
     "The British Admiral who died leading the victory at the Battle of "
     "Trafalgar was"),
    ("What color is historically most associated with communism?",
     "The color historically most associated with communism is"),
    ("What parallel line divided North and South Korea?",
     "North and South Korea were divided by"),
    ("Which Mediterranean island nation did Mussolini's forces bomb extensively "
     "during World War II?",
     "During World War II, the Mediterranean island nation bombed extensively "
     "by Mussolini's forces was"),
    ("Who was the Soviet premier during the Cuban Missile Crisis?",
     "The Soviet premier during the Cuban Missile Crisis was"),
    ("What period of extreme state-sanctioned violence during the French "
     "Revolution was led by Maximilien Robespierre?",
     "The period of extreme state-sanctioned violence during the French "
     "Revolution led by Maximilien Robespierre was"),
    ("Which country launched the first artificial Earth satellite?",
     "The first artificial Earth satellite was launched by"),
    ("Who co-authored 'The Communist Manifesto' with Karl Marx?",
     "'The Communist Manifesto' was co-authored with Karl Marx by"),
    ("What country did the Soviet Union invade in 1979?",
     "In 1979, the Soviet Union invaded"),
    ("Which British Prime Minister coined the phrase 'Iron Curtain'?",
     "The phrase 'Iron Curtain' was coined by British Prime Minister"),
    ("What major river flows directly through the city of Rome?",
     "The major river that flows directly through the city of Rome is"),
    ("What series of three major wars were fought between Rome and Carthage?",
     "The series of three major wars fought between Rome and Carthage was known "
     "as"),
    ("In what year did the Russian Revolution begin?",
     "The Russian Revolution began in the year"),
    ("Which Roman Emperor is famously, though likely mythically, said to have "
     "fiddled while Rome burned?",
     "The Roman Emperor famously, though likely mythically, said to have "
     "fiddled while Rome burned was"),
    ("What was the primary language spoken by the ancient Romans?",
     "The primary language spoken by the ancient Romans was"),
    ("What was the three-word national motto that originated from the French "
     "Revolution?",
     "The three-word national motto that originated from the French Revolution "
     "was"),
    ("What was the primary security agency of the Soviet Union from 1954 to "
     "1991?",
     "From 1954 to 1991, the primary security agency of the Soviet Union was"),
    ("What political philosopher co-authored 'The Communist Manifesto'?",
     "'The Communist Manifesto' was co-authored by the political philosopher"),
    ("What extravagant palace was the primary residence of French kings before "
     "the revolution?",
     "Before the revolution, the extravagant palace that was the primary "
     "residence of French kings was"),
    ("What was the official newspaper of the Italian Fascist Party?",
     "The official newspaper of the Italian Fascist Party was"),
    ("Who was the first American astronaut to orbit the Earth?",
     "The first American astronaut to orbit the Earth was"),
    ("What remote Atlantic island was Napoleon exiled to for the second and "
     "final time?",
     "The remote Atlantic island to which Napoleon was exiled for the second "
     "and final time was"),
    ("Who was the first official Roman Emperor?",
     "The first official Roman Emperor was"),
    ("Which emperor formally split the Roman Empire into eastern and western "
     "halves to make it easier to govern?",
     "The emperor who formally split the Roman Empire into eastern and western "
     "halves, to make it easier to govern, was"),
    ("What ancient empire did Mussolini state he wanted to recreate?",
     "The ancient empire Mussolini stated he wanted to recreate was"),
    ("What execution device became a widespread symbol of the French "
     "Revolution?",
     "The execution device that became a widespread symbol of the French "
     "Revolution was"),
    ("Who was the Prime Minister of Japan during most of World War II?",
     "During most of World War II, the Prime Minister of Japan was"),
]
