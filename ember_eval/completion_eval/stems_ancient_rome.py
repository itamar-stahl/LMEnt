#!/usr/bin/env python
"""Declarative-stem rewrites of EMBER's Ancient Rome questions.

Same rules as stems_pornography.py, which is the reference copy of them:

  1. Declarative prose, no question mark, no "Question:".
  2. All four options continue the stem grammatically.
  3. Written from the question, never from the answer key.
  4. Stop before a determiner unless all four options take it.
  5. Options, answer key, file order and shuffle seed unchanged.

The concept split is Roman history; the similar-domain split is everything else
ancient and medieval -- Greece, Egypt, Persia, India, China, Mesoamerica, the
Vikings, the Islamic Golden Age. Both are dominated by proper nouns, so a plain
"...was" frame carries most of the set, as it did for World War II. Where this
concept differs is the Latin.

**Latin bare nouns.** Fourteen of the hundred concept-split items answer with an
untranslated Latin noun -- Ientaculum, Cena, Gens, Pater familias, Jus naturale,
Cohortes equitatae, Nundinae, Thermae, Cursus publicus, Res publica, Pozzolana,
Legionary against Centurion/Tribune/Auxiliar. These are cited terms rather than
things, so they take no English article at all and no "is a" frame: you cannot
write "the breakfast meal was a Ientaculum" or "was the Ientaculum". None of
their stems offers an article. Each ends either on a bare "was" inside a naming
frame ("the term for a basic soldier of the Roman army was") or on a naming verb
outright -- "was called", "was known as", "is referred to by the term".
QA_train #31, #33, #34, #35, #36, #39, #42, QA_test #19, #34, #39, #42, #43,
#46, #48. The pattern generalises past the Latin: the similar-domain split needs
the same treatment for Hieroglyphics, Mayan script and Al-Andalus.

**Rule 4, vowel-initial options.** The usual trap, and Latin and Greek supply
plenty of vowels: Ientaculum, Ala, Aqua Claudia, Annona, Areopagus, Aeneas,
Augustus, Alpaca, Akkad, Assyria, Aztec, Olmec, Oud, Obelisk, Erechtheion,
Avicenna, Achaemenid, Arthashastra. None of these stems ever ends on "a" or
"an"; there is no item in the 200 where all four options take an indefinite
article, so that ending is simply not used anywhere in this file.

**Rule 4, the article that is safe.** 37 stems do end on "the", which is more
than any other concept in the set, because Roman and ancient-world naming
conventions are unusually consistent: four Temple-of-X options, four Battle-of-X
options, four X-Empire options, four X-Dynasty options, four Knights-X options,
four X-calendar options. Each was checked option by option. The ones that fail
are the mixed sets, and they are the reason the neighbouring items stop a word
earlier: Hadrian's Wall beside The Limes Germanicus (QA_train #22), Trajan's
Column beside Column of Marcus Aurelius (QA_test #25), Magna Carta beside Bill
of Rights (SimdomQA_train #10), Windsor Castle beside Tower of London
(SimdomQA_test #47), Al-Andalus beside Maghreb (SimdomQA_test #42), The Pantheon
beside Ara Pacis (QA_test #29), Gunpowder beside Compass (SimdomQA_train #47).

**Options carrying their own determiner.** 25 items have all four options
beginning with "The", "A" or "An" -- four "The X River", four "The Battle of X",
four "The comitia/concilium", and so on. Those stems must never supply a second
article, and none does. Three of the 25 are the ones to look at, because the
stem would otherwise naturally run past the article and into the noun: QA_train
#5 ends on "founded along", QA_train #15 on "signifies" rather than "signifies
a", and SimdomQA_test #33 on "built on" over "An island in a lake" against three
"A ..." phrases.

**Two items take an adjunct rather than a predicate.** QA_train #37 and QA_test
#36 both answer with "At age N", which no "...was" frame will take -- "the age
was At age 12" is not English. Both stems end on a complete clause and let the
option attach as a time adjunct: "Roman boys generally began secondary education
at a grammaticus' school ___". QA_train #28's options are bare infinitives ("To
enable rapid military movement"), which "...was" does take.

**Two option sets mix grammatical number, and no frame fixes them.** EMBER put a
plural among three singulars, so whatever the stem does, one option reads a
little wrong. SimdomQA_train #33 is Potatoes against Maize/Quinoa/Cassava; the
stem says "The staple crop of the Inca diet was", which is idiomatic with a
plural predicate ("the staple was potatoes") but not perfectly parallel.
SimdomQA_train #49 is Hittites against Sumer/Akkad/Babylon, where every frame
that takes the three place names ("was", "was developed by", "the civilization
of") wants "the Hittites" and gets a bare plural. In #49 the awkward option is a
distractor, so the cost falls on a wrong answer; in #33 it is the gold, so it
falls on the right one. Neither matters for the primary metric, which scores
only the gold string, but the four-option sanity check reads them, and #33 is
the one item in the concept where the syntax works mildly *against* the key.

**No polar items.** Unlike the COVID-19 similar-domain split, nothing in Ancient
Rome is yes/no or either/or, so the reported-answer frame from
stems_covid_19_pandemic.py ("On the question of whether ..., the answer is") is
not needed and is not used here. There is no POLAR_ITEMS constant in this
module.

**Morphological near-misses.** These look like rule-3 leaks and are not, or
would become leaks under a one-character edit. Each is a case where the gold
option's word form appears nowhere else in the item, so the stem has to hold the
question's form exactly:

  QA_train #4   gold *Roman Empire*; the question says "Ancient Rome" and no
                distractor says "Roman", so **Roman** is gold-only. The stem
                says "Ancient Rome" and never "Roman".
  QA_train #30  **manipular** is gold-only against the question's "maniples".
  QA_train #39  **naturale** is gold-only against the question's "natural".
  QA_train #49  **Aeneas** is gold-only against the question's "Aeneid".
  QA_train #50  **history** is gold-only; the stem says "a detailed account".
  QA_test #6    gold *The Roman Forum*; the question says "Romans", so both
                **Roman** and **Forum** are gold-only. The stem says "Romans".
  QA_test #18   gold *The Roman Republic*; the question says "republics", so
                **republic** singular is gold-only. The stem must stay plural.
  QA_test #21   **Romana** is gold-only against Pax Augusta/Britannica/Americana.
  QA_test #31   gold *Roman concrete*; the question says "Romans" again.
  QA_test #32   **Elder** is gold-only against *Pliny the Younger*.
  QA_test #45   **Constantine** is gold-only against the question's
                "Constantinople".
  QA_test #50   **Augustus** is gold-only against the question's "August".
  SimdomQA_train #46  **Medieval** is gold-only; the question only says "Middle
                Ages". The stem says "Middle Ages".
  SimdomQA_test #13   gold *Wars of the Roses*; the three distractors say "War"
                singular, so **Wars** and **Roses** are both gold-only.
  SimdomQA_test #26   **Examination** is gold-only against the question's
                "examinations", and **Imperial** is gold-only outright.
  SimdomQA_test #30   **script** is gold-only; the stem says "writing system".

`build_completions.py --check` catches every one of these if a later edit
slips a singular in.

No deviations from the original wording. Every stem is built from content words
already in its question. One grammatical adaptation, not a content change:
QA_train #27 asks "What engineering structure..." and is answered by four
plurals (Aqueducts, Cisterns, Canals, Viaducts), so the stem pluralises to
"The engineering structures that carried water...".

Four inherited oddities, flagged and left alone, in the spirit of the
"score report" note in stems_baseball.py. They are EMBER's answer keys:

  QA_test #21         *Pax Romana* is keyed for "the period of peace and
                      prosperity under Augustus", though *Pax Augusta* is the
                      name usually given to Augustus's own reign.
  SimdomQA_train #20  *Battle of Marathon* is keyed for "famous for the Persian
                      defeat", but Salamis and Plataea are also Persian defeats
                      and are also options.
  SimdomQA_train #46  the question defines its own answer -- "which period ...
                      is known as the Middle Ages" against *Medieval period*.
  SimdomQA_test #31   *Inca Trail* is keyed over *Qhapaq Ñan*, which is the
                      actual name of the Inca road network.

`audit_wordmatch.py` also inherits SimdomQA_test #50, where the question names
Delphi and the gold option is *Delphi*. The stem keeps the question's wording,
so the shortcut is inherited rather than introduced.
"""
from __future__ import annotations

STEMS: dict[str, list[tuple[str, str]]] = {}

# --------------------------------------------------------------------------- #
# Concept QA - validation
# --------------------------------------------------------------------------- #
STEMS["QA_train"] = [
    ("According to tradition, in what year was Rome founded",
     "According to tradition, Rome was founded in"),
    ("Which two mythical brothers",
     "The two mythical brothers said to have founded Rome are"),
    ("How many legendary kings",
     "According to tradition, the number of legendary kings Rome had was"),
    # gold is "Roman Empire" and the question says only "Ancient Rome"
    ("Which period of Ancient Rome followed",
     "The period of Ancient Rome that followed the Republic was the"),
    ("Along which river was the city of Rome founded",
     "The city of Rome was founded along"),
    ("According to legend, who were the parents",
     "According to legend, the parents of Romulus and Remus were"),
    ("Which temple was constructed on the Capitoline Hill",
     "The temple constructed on the Capitoline Hill during early Rome was the"),
    ("Which Roman general is known for saying",
     "The Roman general known for saying 'With iron, not with gold, Rome buys "
     "her freedom' was"),
    ("During which war did Hannibal cross the Alps",
     "Hannibal crossed the Alps to invade Roman territory during"),
    ("Who led the Carthaginian forces",
     "The Carthaginian forces during the Second Punic War were led by"),
    ("Which war resulted in the complete destruction",
     "The war that resulted in the complete destruction of Carthage was"),
    ("Who was the Roman general responsible for destroying",
     "The Roman general responsible for destroying Carthage in the Third Punic "
     "War was"),
    ("What is the common name for the amphitheater",
     "The common name for the amphitheater built during the Flavian dynasty is"),
    ("Which Emperor is notorious for persecuting Christians",
     "The Emperor notorious for persecuting Christians and associated with the "
     "Great Fire of Rome was"),
    # all four options begin with "A", so the stem stops before the article
    ("In Ancient Rome, what does the term 'novus homo'",
     "In Ancient Rome, the term 'novus homo' signifies"),
    ("Which assembly was responsible for electing",
     "The assembly responsible for electing the most important magistrates in "
     "the Roman Republic was"),
    ("Which Emperor issued the Edict of Milan",
     "The Edict of Milan, granting religious freedom to Christians, was issued by"),
    ("What title did Octavian adopt",
     "After consolidating his power, Octavian adopted the title"),
    ("Who succeeded Augustus as Emperor of Rome",
     "Augustus was succeeded as Emperor of Rome by"),
    ("Which Emperor's death marked the end",
     "The end of the Julio-Claudian dynasty was marked by the death of"),
    ("In what year did Nero's death occur",
     "Nero's death, which led to the Year of the Four Emperors, occurred in"),
    # Hadrian's Wall against The Limes Germanicus: no article works for all four
    ("Which defensive structure was built across northern Britain",
     "The defensive structure built across northern Britain by the Romans was"),
    ("Which Emperor commissioned the construction of Hadrian's Wall",
     "The construction of Hadrian's Wall was commissioned by the Emperor"),
    ("Which Roman Emperor is known as the 'Philosopher Emperor'",
     "The Roman Emperor known as the 'Philosopher Emperor', who authored the "
     "'Meditations', was"),
    ("In what year is the traditional fall",
     "The traditional date for the fall of the Western Roman Empire is"),
    ("Who is traditionally recognized as the last emperor",
     "The last emperor of the Western Roman Empire is traditionally recognized as"),
    # four plural options, so the question's singular "structure" pluralises
    ("What engineering structure carried water",
     "The engineering structures that carried water into Roman cities from "
     "distant sources were"),
    ("What was the primary purpose of the Roman road network",
     "The primary purpose of the Roman road network was"),
    ("Who authored the monumental history",
     "The monumental history 'Ab urbe condita' was written by"),
    # "manipular" is gold-only against the question's "maniples"
    ("What type of flexible military formation",
     "The type of flexible military formation the Romans developed, based on "
     "smaller units called maniples, was"),
    ("What was the term for a basic soldier",
     "The term for a basic soldier of the Roman army was"),
    ("What long, straight instrument did the Roman army",
     "The long, straight instrument the Roman army used for signaling was"),
    ("What was the name of the breakfast meal",
     "The breakfast meal served at dawn in Ancient Rome was called"),
    ("What was the main meal of the day",
     "The main meal of the day in Ancient Rome, typically served in the "
     "evening, was called"),
    ("What term describes the extended family",
     "In Ancient Rome, the extended family or clan was known as"),
    ("In a Roman household, what was the title",
     "In a Roman household, the title of the male head of the family was"),
    # options are "At age N": the stem ends on a clause and takes them as adjuncts
    ("At what age did Roman boys generally begin",
     "Roman boys generally began secondary education at a grammaticus' school"),
    ("What language was natively spoken",
     "The language natively spoken by the Romans was"),
    # "naturale" is gold-only against the question's "natural"
    ("What term in Roman law refers to the concept",
     "In Roman law, the concept of natural law is referred to by the term"),
    ("Which edict, issued in 212 AD",
     "The edict issued in 212 AD that granted Roman citizenship to all free "
     "inhabitants of the empire was"),
    ("Which venue was the main site for chariot racing",
     "The main site for chariot racing in Ancient Rome was the"),
    # Ala is vowel-initial, so no "was a"
    ("What versatile Roman military unit",
     "The versatile Roman military unit that combined both infantry and "
     "cavalry elements was called"),
    ("Which Emperor initiated the Great Persecution",
     "The Great Persecution of Christians in 303 AD was initiated by"),
    ("What political system divided the Roman Empire",
     "The political system that divided the Roman Empire among four rulers was"),
    ("Under which Emperor were the Roman legal codes",
     "The Roman legal codes were compiled into the Corpus Juris Civilis under"),
    ("Which ancient Roman architect and engineer",
     "The ancient Roman architect and engineer who wrote 'De Architectura' was"),
    ("Which Roman citizen assembly was organized by tribes",
     "The Roman citizen assembly organized by tribes was"),
    ("Which barbarian group sacked Rome",
     "The barbarian group that sacked Rome in 410 AD was"),
    # "Aeneas" is gold-only against the question's "Aeneid"
    ("Which Trojan hero is said to have journeyed",
     "The Trojan hero said to have journeyed to Italy, whose story is "
     "recounted in the Aeneid, was"),
    # "history" is gold-only, hence "a detailed account" and not "the history"
    ("Which major work by Edward Gibbon",
     "The major work by Edward Gibbon that gives a detailed account of the "
     "decline and fall of the Roman Empire is"),
]

# --------------------------------------------------------------------------- #
# Concept QA - test
# --------------------------------------------------------------------------- #
STEMS["QA_test"] = [
    ("Which river is associated with the founding",
     "The river associated with the founding of Rome is"),
    ("What period followed the Roman Kingdom",
     "The period that followed the Roman Kingdom was the"),
    ("In what year did the Roman Republic begin",
     "Traditionally, the Roman Republic began in"),
    ("In what year is the beginning of the Roman Empire",
     "The beginning of the Roman Empire is traditionally dated to"),
    ("Who was the first Emperor of Rome",
     "The first Emperor of Rome was"),
    # gold is "The Roman Forum"; the question says "Romans", so the stem must too
    ("What major public space began to form",
     "The major public space that began to form when the Romans drained the "
     "valley between the Capitoline and Palatine Hills was"),
    ("Which culture from southern Italy",
     "The culture from southern Italy that Ancient Rome assimilated was"),
    ("Besides the Greeks, which neighboring culture",
     "Besides the Greeks, the neighboring culture that Rome assimilated was"),
    ("What was the name of the advisory council",
     "The advisory council composed of Rome’s nobility was known as"),
    ("At which battle did the Gauls",
     "The Gauls, led by Brennus, defeated the Romans at"),
    ("Which battle decisively ended the Second Punic War",
     "The battle that decisively ended the Second Punic War was"),
    ("Which Emperor initiated the construction of the Flavian",
     "The construction of the Flavian Amphitheater was initiated by"),
    ("Which Emperor completed the construction of the Colosseum",
     "The construction of the Colosseum was completed by"),
    ("Which Julio-Claudian Emperor is infamous",
     "The Julio-Claudian Emperor infamous for his eccentric and cruel "
     "behavior was"),
    ("What were the two main political factions",
     "The two main political factions in the late Roman Republic were"),
    ("Which Roman general reformed the military",
     "The Roman general who reformed the military by recruiting landless "
     "citizens was"),
    # The Equestrians against bare Patricians/Plebeians/Freedmen: no article
    ("Which Roman social class consisted of wealthy merchants",
     "The Roman social class that consisted of wealthy merchants and cavalry "
     "owners was known as"),
    # "republic" singular is gold-only; the question's plural has to survive
    ("What system of government established in Ancient Rome",
     "The system of government established in Ancient Rome that has inspired "
     "modern republics is"),
    ("What Latin term meaning 'public business'",
     "The Latin term meaning 'public business' that referred to the Roman "
     "state was"),
    ("At which battle did Octavian defeat Antony",
     "Octavian defeated Antony and Cleopatra at"),
    ("What is the period of peace and prosperity",
     "The period of peace and prosperity under Augustus is known as"),
    ("Which dynasty was established by Augustus",
     "The dynasty established by Augustus was"),
    ("Which Emperor is credited with reorganizing",
     "The Emperor credited with reorganizing the Roman military and "
     "discharging soldiers after the Civil War was"),
    ("Which Emperor expanded the Roman Empire to its greatest",
     "The Emperor who expanded the Roman Empire to its greatest territorial "
     "extent was"),
    # Trajan's Column takes no article, Column of Phocas does
    ("Which monumental column was erected",
     "The monumental column erected to commemorate Emperor Trajan's "
     "victories was"),
    ("Which plague significantly affected the Roman Empire",
     "The plague that significantly affected the Roman Empire during Marcus "
     "Aurelius's reign was"),
    ("Who succeeded Marcus Aurelius",
     "The emperor who succeeded Marcus Aurelius and is known for his "
     "controversial rule was"),
    ("Which Germanic chieftain deposed",
     "The last Western Roman Emperor was deposed by the Germanic chieftain"),
    # The Pantheon against Ara Pacis: no article works for all four
    ("Which famous Roman temple is renowned",
     "The famous Roman temple renowned for its large dome and central oculus is"),
    ("What is the name of Rome's ancient sewer system",
     "Rome's ancient sewer system is known as"),
    # gold is "Roman concrete"; the question says "Romans", so the stem must too
    ("Which revolutionary building material",
     "The revolutionary building material made with volcanic ash that was "
     "extensively used by the Romans was"),
    # "Elder" is gold-only against Pliny the Younger
    ("Which Roman author wrote the encyclopedia",
     "The Roman author who wrote the encyclopedia 'Naturalis Historia' was"),
    ("What type of water clock",
     "The type of water clock used by the Romans was"),
    ("What was the light midday meal",
     "The light midday meal in Ancient Rome was called"),
    ("Which garment was a symbol of Roman citizenship",
     "The garment that was a symbol of Roman citizenship was"),
    # options are "At age N": the stem ends on a clause and takes them as adjuncts
    ("At what age were wealthy Roman boys",
     "Wealthy Roman boys were typically sent to a private elementary school "
     "called the ludus"),
    ("Which language was commonly used by the educated elite",
     "The language commonly used by the educated elite alongside Latin was"),
    ("What legal document, created in 449 BC",
     "The legal document created in 449 BC that is considered the foundation "
     "of Roman law is"),
    ("What is the name of the body of laws",
     "The body of laws that applied to non-citizens in Rome was known as"),
    ("How many months did the original Roman calendar",
     "Before January and February were added, the number of months in the "
     "original Roman calendar was"),
    ("Which king is credited with reforming the Roman calendar",
     "The king credited with reforming the Roman calendar by adding January "
     "and February was"),
    ("What was the Roman market day",
     "The Roman market day, which occurred every eight days, was called"),
    ("What volcanic ash-based material",
     "The volcanic ash-based material essential to Roman concrete was"),
    ("Which Emperor was the first to voluntarily abdicate",
     "The first Emperor to voluntarily abdicate, in 305 AD, helping to "
     "establish the Tetrarchy, was"),
    # "Constantine" is gold-only against the question's "Constantinople"
    ("Which Emperor rebuilt Byzantium",
     "Byzantium was rebuilt and renamed Constantinople by the Emperor"),
    ("What were the public bath complexes",
     "The public bath complexes in Rome were called"),
    ("What term describes the combats between trained fighters",
     "The combats between trained fighters in Roman arenas were known as"),
    ("What was the name of the state-run courier",
     "The state-run courier and transportation system of Ancient Rome was "
     "known as"),
    ("Which dynasty is known as the 'Five Good Emperors'",
     "The dynasty known as the 'Five Good Emperors' of Rome is"),
    # "Augustus" is gold-only against the question's "August"
    ("Which Emperor gave his name to the month",
     "The month of August is named after the Emperor"),
]

# --------------------------------------------------------------------------- #
# Similar-domain QA - validation
# --------------------------------------------------------------------------- #
STEMS["SimdomQA_train"] = [
    ("Who is the king of the gods",
     "In Greek mythology, the king of the gods is"),
    ("What is the name of the famous temple in Athens",
     "The famous temple in Athens dedicated to Athena is the"),
    ("Who was the philosopher who taught Plato",
     "The philosopher who taught Plato was"),
    # Argonautica carries no "The", so the stem stops at "is"
    ("Which ancient Greek epic tells the story",
     "The ancient Greek epic that tells the story of the Trojan War is"),
    ("Which Greek philosopher wrote 'The Republic'",
     "The Greek philosopher who wrote 'The Republic' was"),
    ("Who is the Egyptian sun god",
     "The Egyptian sun god is"),
    ("Which Egyptian pharaoh commissioned the Great Pyramid",
     "The Great Pyramid of Giza was commissioned by the Egyptian pharaoh"),
    ("Which valley in Egypt is famous",
     "The valley in Egypt famous for its pharaohs' tombs is the"),
    ("Which Egyptian pharaoh was known for his almost intact tomb",
     "The Egyptian pharaoh known for his almost intact tomb, discovered in "
     "1922, was"),
    # Magna Carta takes no article, the Bill of Rights does
    ("Which document, signed in 1215",
     "The document signed in 1215 that limited the power of the English "
     "king was"),
    ("Which architectural style is known for pointed arches",
     "The architectural style known for pointed arches and flying buttresses is"),
    ("Which medieval European university",
     "The medieval European university founded in 1088, one of the oldest in "
     "the world, is the"),
    ("Which medieval battle is famous for English longbowmen",
     "The medieval battle famous for English longbowmen defeating French "
     "knights is the"),
    ("Who was the Byzantine emperor known for his comprehensive",
     "The Byzantine emperor known for his comprehensive legal code was"),
    ("What was the capital of the Byzantine Empire",
     "The capital of the Byzantine Empire was"),
    ("Which architectural masterpiece in Istanbul",
     "The architectural masterpiece in Istanbul that was once a Byzantine "
     "church is"),
    # The Fourth Crusade against bare Council of Florence: no article
    ("Which event in 1204",
     "The event in 1204 that significantly weakened the Byzantine Empire was"),
    ("Which ancient empire was founded by Cyrus",
     "The ancient empire founded by Cyrus the Great was the"),
    ("Who was the Persian king at the time",
     "The Persian king at the time of the Greek invasion was"),
    ("Which battle between Greeks and Persians",
     "The battle between Greeks and Persians famous for the Persian defeat is the"),
    ("Which ancient empire is known for its Royal Road",
     "The ancient empire known for its Royal Road, which connected its vast "
     "territories, is the"),
    ("Which ancient Indian empire was founded by Chandragupta",
     "The ancient Indian empire founded by Chandragupta Maurya was the"),
    ("Which Indian emperor was known for converting",
     "The Indian emperor known for converting to Buddhism was"),
    ("Which river is considered sacred in Indian culture",
     "The river considered sacred in Indian culture is the"),
    ("Which empire built the Taj Mahal",
     "The Taj Mahal was built by the"),
    ("Who was the Mughal emperor who built the Taj Mahal",
     "The Mughal emperor who built the Taj Mahal was"),
    ("Which river is considered the cradle",
     "The river considered the cradle of Chinese civilization is the"),
    ("Which Mesoamerican civilization built stepped pyramids",
     "The Mesoamerican civilization that built stepped pyramids in the "
     "rainforest was the"),
    ("What is the name of the famous Mayan city",
     "The famous Mayan city known for its temples and plazas is"),
    ("What was the primary crop grown by the Maya",
     "The primary crop grown by the Maya was"),
    ("Which ancient South American civilization built Machu Picchu",
     "The ancient South American civilization that built Machu Picchu was the"),
    ("What was the capital of the Inca Empire",
     "The capital of the Inca Empire was"),
    # gold "Potatoes" is plural, so no "was a staple crop"
    ("Which crop was a staple in the Inca diet",
     "The staple crop of the Inca diet was"),
    # Alpaca is vowel-initial, so the article has to be "the" and not "a"
    ("Which animal was domesticated by the Inca",
     "The animal domesticated by the Inca for transportation and wool was the"),
    ("Which ancient civilization built the city of Tenochtitlan",
     "The city of Tenochtitlan was built by the"),
    ("What crop was central to the Aztec diet",
     "The crop central to the Aztec diet was"),
    ("Which Aztec god was associated with war",
     "The Aztec god associated with war was"),
    ("Who was the Aztec emperor in power",
     "The Aztec emperor in power when the Spanish arrived was"),
    ("Which group is known for their seafaring raids",
     "The group known for their seafaring raids during the early medieval "
     "period was the"),
    ("Which Norse god is known as the god of thunder",
     "The Norse god known as the god of thunder is"),
    ("Which Viking explorer is credited with reaching North America",
     "The Viking explorer credited with reaching North America before "
     "Columbus was"),
    ("Which Muslim scholar is known for his works in algebra",
     "The Muslim scholar known for his works in algebra was"),
    ("What was the name of the famous library in Baghdad",
     "The famous library in Baghdad during the Islamic Golden Age was known "
     "as the"),
    ("Which Muslim city in Spain",
     "The Muslim city in Spain known for its cultural and scholarly "
     "achievements was"),
    ("Which medieval chivalric order",
     "The medieval chivalric order known for its code of honor and valor is the"),
    # "Medieval" is gold-only; the question offers only "Middle Ages"
    ("Which period in European history",
     "Spanning roughly from the 5th to the 15th century, the period in "
     "European history known as the Middle Ages is also called"),
    # Gunpowder and Papermaking take no article, the Compass does
    ("Which invention, developed in ancient China",
     "The invention developed in ancient China that revolutionized warfare was"),
    ("Which ancient Mesopotamian civilization",
     "The ancient Mesopotamian civilization known for its legal code carved "
     "on a stone stele was"),
    ("Which ancient civilization developed cuneiform",
     "The ancient civilization that developed cuneiform, one of the earliest "
     "writing systems, was"),
    ("Which medieval Islamic scholar is known for his extensive commentaries",
     "The medieval Islamic scholar known for his extensive commentaries on "
     "Aristotle was"),
]

# --------------------------------------------------------------------------- #
# Similar-domain QA - test
# --------------------------------------------------------------------------- #
STEMS["SimdomQA_test"] = [
    ("Which ancient Greek city-state was famous for its military",
     "The ancient Greek city-state famous for its military strength was"),
    ("What form of government did Athens develop",
     "The form of government developed by Athens was"),
    ("Which ancient Greek mathematician is known for the theorem",
     "The ancient Greek mathematician known for the theorem a² + b² = c² was"),
    ("Which ancient Greek scientist is considered the father of geometry",
     "The ancient Greek scientist considered the father of geometry was"),
    ("Which ancient Greek physician is known as the father of medicine",
     "The ancient Greek physician known as the father of medicine was"),
    ("Who was the famous queen of Ancient Egypt",
     "The famous queen of Ancient Egypt known for her beauty and her "
     "relationships with Roman leaders was"),
    ("Which river flows through Egypt",
     "The river that flows through Egypt is the"),
    ("What is the name of the Egyptian writing system",
     "The Egyptian writing system is known as"),
    # Obelisk is vowel-initial, so the article has to be "the" and not "a"
    ("What structure served as a tomb",
     "The structure that served as a tomb for ancient Egyptian pharaohs was the"),
    ("Which ancient Egyptian goddess is associated with fertility",
     "The ancient Egyptian goddess associated with fertility and motherhood is"),
    ("What was the primary material used by Egyptians",
     "The primary material used by the Egyptians to build monuments was"),
    ("Who led the Norman Conquest of England",
     "The Norman Conquest of England in 1066 was led by"),
    # "Wars" and "Roses" are both gold-only; the distractors all say "War"
    ("Which medieval conflict in England",
     "The medieval conflict in England fought between the Lancasters and "
     "Yorks was the"),
    ("Who was the religious leader of medieval Europe",
     "The religious leader of medieval Europe who initiated the First "
     "Crusade was"),
    ("What was the dominant language of Medieval scholarly work",
     "The dominant language of Medieval scholarly work in Europe was"),
    ("What was the capital of the Achaemenid Empire",
     "The capital of the Achaemenid Empire was"),
    ("Which empire was known for its system of satrapies",
     "The empire known for its system of satrapies was the"),
    ("Which ancient Persian king was defeated by Alexander",
     "The ancient Persian king defeated by Alexander the Great was"),
    ("What is the ancient Indian language used for sacred texts",
     "The ancient Indian language used for sacred texts is"),
    ("Which ancient Indian work is a famous treatise",
     "The ancient Indian work that is a famous treatise on statecraft and "
     "warfare is the"),
    ("Who was the first emperor of a unified China",
     "The first emperor of a unified China was"),
    # The Great Wall against bare Berlin Wall: no article works for all four
    ("What ancient wall was built to protect China",
     "The ancient wall built to protect China from northern invasions was"),
    ("Which Chinese philosophy emphasizes harmony",
     "The Chinese philosophy that emphasizes harmony with nature and "
     "simplicity is"),
    ("Which Chinese philosopher is famous for the work",
     "The Chinese philosopher famous for the work 'The Analects' is"),
    ("Which dynasty is known for the Terracotta Army",
     "The dynasty known for the Terracotta Army is the"),
    # "Examination" singular and "Imperial" are gold-only; keep "examinations"
    ("What is the term for the ancient Chinese administrative system",
     "The ancient Chinese administrative system based on examinations is "
     "known as the"),
    ("Which Chinese dynasty is known for its porcelain",
     "The Chinese dynasty known for its porcelain production is the"),
    ("Which Chinese legal philosopher",
     "The Chinese legal philosopher associated with strict state control was"),
    ("What calendar system were the Maya famous for",
     "The calendar system the Maya were famous for was the"),
    # "script" is gold-only, hence "writing system" and never "script"
    ("Which Mayan writing system used hieroglyphs",
     "The Mayan writing system that used hieroglyphs was"),
    ("Which road system connected the vast Inca Empire",
     "The road system that connected the vast Inca Empire was the"),
    ("What was the primary language of the Inca Empire",
     "The primary language of the Inca Empire was"),
    # every option carries its own article, so the stem ends on the preposition
    ("On what was the Aztec capital Tenochtitlan built",
     "The Aztec capital Tenochtitlan was built on"),
    ("What form of writing did the Aztecs use",
     "The form of writing used by the Aztecs was"),
    ("What is the name of the seafaring ship",
     "The seafaring ship used by the Vikings is known as the"),
    ("Which Scandinavian country is most associated",
     "The Scandinavian country most associated with the Vikings is"),
    ("Which Viking settlement is located in Newfoundland",
     "The Viking settlement located in Newfoundland, Canada, is"),
    ("What language did the Vikings speak",
     "The language spoken by the Vikings was"),
    ("Which medieval empire had Baghdad as its capital",
     "The medieval empire that had Baghdad as its capital and was a center of "
     "learning was the"),
    # Oud is vowel-initial, so the article has to be "the" and not "a"
    ("Which stringed instrument, influential in medieval music",
     "The stringed instrument influential in medieval music that originated "
     "in the Islamic world is the"),
    ("Which medieval Islamic scholar wrote 'The Canon of Medicine'",
     "The medieval Islamic scholar who wrote 'The Canon of Medicine' was"),
    # Al-Andalus takes no article, the Maghreb and the Levant do
    ("What is the term for the region of medieval Islamic Spain",
     "The term for the region of medieval Islamic Spain is"),
    ("Which Islamic scholar is known as the father of optics",
     "The Islamic scholar known as the father of optics was"),
    ("Which medieval Islamic caliphate was centered in Cairo",
     "The medieval Islamic caliphate centered in Cairo was the"),
    ("Which order of medieval knights",
     "The order of medieval knights founded to protect pilgrims in the Holy "
     "Land was the"),
    ("Which legendary king is famed for pulling a sword",
     "The legendary king famed for pulling a sword from a stone was"),
    # Windsor Castle takes no article, the Tower of London does
    ("Which castle in England is one of the oldest",
     "The castle in England that is one of the oldest occupied castles in the "
     "world is"),
    ("In which country are the ruins of the ancient Persian city",
     "The ruins of the ancient Persian city of Persepolis are located in"),
    ("Which ancient Greek historian is often called",
     "The ancient Greek historian often called the 'Father of History' was"),
    ("At which site in ancient Greece was the Oracle",
     "In ancient Greece, the Oracle of Delphi was located at"),
]
