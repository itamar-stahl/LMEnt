#!/usr/bin/env python
"""Declarative-stem rewrites of EMBER's Cannabis questions.

Same rules as stems_pornography.py, which is the reference copy of them:

  1. Declarative prose, no question mark, no "Question:".
  2. All four options continue the stem grammatically.
  3. Written from the question, never from the answer key.
  4. Stop before a determiner unless all four options take it.
  5. Options, answer key, file order and shuffle seed unchanged.

The concept split is botany, chemistry, slang and drug law; the similar-domain
split is every *other* recreational substance - alcohol, tobacco, caffeine,
opioids, stimulants, psychedelics, depressants and inhalants. Neither split
contains a single yes/no or either/or question, so the reported-answer frame
that stems_covid_19_pandemic.py needed ("On the question of whether ..., the
answer is ___") is used **nowhere** in this file. All 200 stems are ordinary
noun-phrase or predicate frames.

**Rule 4.** This is the concept the rule was written for. Cannabis and drug
vocabulary is saturated with vowel-initial terms, and a single one among the
four options kills an "...is a" ending for the whole item: Edible, Elixir,
Entourage, Ethanol, Isopropyl alcohol, Oxygen, Argon, Arrhythmia, Addiction,
Ocimene, Alkaloids, Uruguay, Epidiolex, Industrial hemp, Ivy, Air layering,
Egypt, India, Anorexia, Osteoporosis, Autoflowering seeds, Algae, Oxidation,
Edibles, Indigestion, Euphoria, Anhedonia, Ataxia, Alkanes, Amsterdam,
Olive oil, Acapulco Gold, Anemometer, Altimeter, International Units, Ale,
Inca Kola, Espresso, Americano, Iced latte, Oxycodone, OxyContin, Opium resin,
Oxymorphone, Amphetamine, Atomoxetine, Ephedrine, Oxymetazoline, Armodafinil,
Epinephrine, Orlistat, Ibotenic acid, Ergotamine, Ayahuasca, Ibogaine,
Opioids, Alprazolam, Antipsychotics, Eszopiclone, Amobarbital, Isopropanol,
On the rocks, Acetaldehyde, Electronic cigarette, Opium, Ice, Isobutane,
Acetone, Octane, Aerosol propellants, Amyl nitrite, Ether. 77 of the 200 items
have at least one vowel-initial option, so the default here is to stop at "is",
"is called" or "are called" and let the option supply its own article.

Plurals break it a second way. 37 items have an option ending in a plural -s,
and ten of those are all-plural sets: QA_train #22, #46, #48, QA_test #20, #25,
#41, SimdomQA_train #40, #42, SimdomQA_test #39, #48. QA_train #6
(Trichomes / Stomata / Pistils / Calyxes) is an eleventh in everything but the
spelling of Stomata. Those stems either end in "are called" / "known as"
(QA_train #22, #46, SimdomQA_train #40, SimdomQA_test #39) or use a naming
frame whose singular subject is the *term*, not the referent, so that "is
called" agrees with the subject and never with the option (QA_train #48,
QA_test #20, #25, SimdomQA_train #42, SimdomQA_test #48). QA_test #41
(Seeds / Leaves / Flowers / Stems) takes a possessive frame,
"...produced from the cannabis plant's ___", for the same reason: "the part
... is Seeds" agrees with nothing.

Nineteen stems do end in a determiner, and each was checked by hand against
all four of its options:

  "...is the"        QA_train #4, QA_test #10 (four X receptor names),
                     QA_test #40 (four gram-sized units),
                     SimdomQA_test #31 (four seeds and nuts)
  "...was the"       QA_train #23, #24, QA_test #29, #31 (four statute and
                     treaty names, all of which carry "the")
  "...is called the" QA_train #5 (four named effects)
  "...is a" / "is called a"
                     QA_train #7, #8, #14, #43, QA_test #15,
                     SimdomQA_train #7, #8, #9, #10, SimdomQA_test #14
                     (all-consonant sets: smoking formats, containers,
                     tobacco products)

**Options that are not noun phrases.** One item, SimdomQA_test #7, has
On the rocks / Neat / Straight up / Chilled - a prepositional phrase, two
adjectives and a particle verb, which no ordinary subject-predicate frame
takes. It uses the question's own naming frame, "The term for an alcoholic
drink served over ice is ___", where any string reads as the term being named.
That is the same device stems_baseball.py used for its clause-valued options,
and the clumsiness falls on all four equally.

**Questions whose phrasing forced an unusual frame.** Five similar-domain
questions name their own answer inside a predicate slot rather than in the
wh-slot, so the faithful transposition puts the blank where that name was and
loses nothing:

  SimdomQA_train #18  "...and is commonly sold as Vicodin?"
                      -> "...combines hydrocodone with acetaminophen is
                         commonly sold as ___"
  SimdomQA_train #30  "...and is known as modafinil?"     -> "...is known as ___"
  SimdomQA_train #32  "...and known as phentermine?"      -> "...is known as ___"
  SimdomQA_train #33  "..., is BZP?"                      -> "...substitute is ___"
  SimdomQA_train #39  "...and contains ibogaine?"         -> "...practices contains ___"

**One deviation from the original wording**, following the precedent set by the
Cittagazze spectres item in stems_harry_potter.py:

  SimdomQA_train #48 "Which prescription medication, gabapentin, is sometimes
      used off-label for anxiety but primarily for nerve pain?" drops the
      appositive **gabapentin**. Here the answer sits in an appositive rather
      than in a predicate slot, so there is no blank for it to vacate and any
      stem that keeps it reads "...for nerve pain, gabapentin, is Gabapentin".
      Reviewer's note: closing this one costs something. Pregabalin, a
      distractor, also fits "off-label for anxiety, primarily for nerve pain",
      so the item goes from tautological to contestable rather than to clean.
      It is the only wording change in the file.

**Fourteen items answer themselves and are left alone**, exposed below as
SELF_ANSWERING_ITEMS. In each of these EMBER's own question already contains
the gold string, so any faithful stem contains it too, and unlike the five
above there is no predicate slot for the blank to occupy: removing the phrase
does not make a harder item, it makes an unanswerable or ambiguous one. They
are EMBER's items and the multiple-choice runs already recorded scored the same
tautology, so they stay in place and item N here is item N there. Filter them
at analysis time, exactly as stems_covid_19_pandemic.py recommends for its
polar items:

    from stems_cannabis import SELF_ANSWERING_ITEMS
    rows = [r for r in results["results"]
            if r["question"] not in SELF_ANSWERING_ITEMS]

The worst of them is QA_train #40, "What material made from hemp fibers was
historically used for making sails and rope?", whose key is *Rope* over
*Canvas*. Dropping "and rope" would make the item read for Canvas and fight
the answer key, which is the Tobruk case in stems_world_war_ii.py in reverse,
so the premise is kept hint and all.

**Morphological near-misses.** These look like leaks and are not; each stem
must keep the question's form of the word and never drift to the gold's, and
build_completions.py --check catches it if a later edit slips:

  QA_train #14   "dabbing", never "dab"        (gold *Dab rig*)
  QA_test  #3    "flowers", never "flowering"  (gold *Flowering plant*)
  QA_test  #27   "pine", never "pinene"        (gold *Pinene*)
  QA_test  #44   "coconut", never "coco"       (gold *Coco coir*)
  QA_test  #48   "cartridges", never "cartridge" (gold *Vape cartridge*)
  QA_test  #50   "dried", never "dry"          (gold *Dry sift*)
  SimdomQA_train #11  "kola", never "cola"     (gold *Cola*)
  SimdomQA_train #24  "coca", never "cocaine"  (gold *Cocaine*)
  SimdomQA_train #28  "ephedra", never "ephedrine" (gold *Ephedrine*)
  SimdomQA_test  #11  "chew", never "chewing"  (gold *Chewing tobacco*)
  SimdomQA_test  #26  "deter", never "deterrent" (gold *Abuse-deterrent formulation*)

Two single-character traps are worth naming, because the leak checker tokenizes
on [a-z0-9]+ and a lone letter is a content word. QA_test #28's gold is
*Schedule I* against Schedule II / III / V, so a bare "i" token anywhere in
that stem would leak; QA_test #23's gold is *T-break*, so a bare "t" would.
Neither stem contains one.

Two inherited oddities, kept because they are equally wrong for all four
options and so favour none of them, which is the "northern European" case in
stems_pornography.py: QA_test #8 and #9 both ask for a "two-letter
abbreviation" whose answer is the three-letter THC or CBD, and QA_train #40's
premise (above) names the answer. Both miscues apply to every option in their
set, so neither favours one.
"""
from __future__ import annotations

# --------------------------------------------------------------------------- #
# The fourteen items whose question already contains its own answer.
#
# Each entry below is also marked inline with `# SELF-ANSWERING`. These are
# EMBER's circularities, not ones the rewrite introduced, and they are kept for
# the reason given in the docstring: there is no blank for the answer phrase to
# vacate, so deleting it produces an ambiguous item rather than a harder one.
#
# Do not delete the entries to exclude them. build_completions.py requires
# exactly 50 stems per split and renumbering would break the guarantee that
# item N here is item N in every multiple-choice run already recorded. Filter
# on the `question` field at analysis time instead.
#
# Expect these fourteen to be near-ceiling for every model, ablated or not, so
# they compress the specificity side of the h-score. Eight of the fourteen sit
# in the concept splits and six in the similar-domain splits.
# --------------------------------------------------------------------------- #
SELF_ANSWERING_ITEMS: frozenset[str] = frozenset({
    "What solvent is commonly used to make butane hash oil (BHO)?",
    "What is the term for cannabis products used to relieve medical conditions?",
    "What material made from hemp fibers was historically used for making sails and rope?",
    "What is the botanical genus of cannabis?",
    "What name is given to the sticky wax-like cannabis concentrate?",
    "What act legalized recreational cannabis in Canada in 2018?",
    "What part of the cannabis plant is used to produce hemp seeds for nutrition?",
    "What nutrient-rich hemp seed product is also known as hemp hearts?",
    "Which beverage is prepared by steeping tea leaves in hot water?",
    "Which plant-based soda popular in Brazil is made from guaraná seeds and contains caffeine?",
    "Which energy drink is marketed with the slogan 'Red Bull gives you wings'?",
    "Which synthetic opioid is known by the brand name Fentanyl?",
    "What is the term for a blunt wrap filled with cannabis?",
    "Which synthetic psychedelic phenethylamine is also known by the code 2C-B?",
})

STEMS: dict[str, list[tuple[str, str]]] = {}

# --------------------------------------------------------------------------- #
# Concept QA - validation
# --------------------------------------------------------------------------- #
STEMS["QA_train"] = [
    ("Which species of cannabis is known for its short stature",
     "The species of cannabis known for its short stature and broad leaves is"),
    ("Which species of cannabis is known for its auto-flowering",
     "The species of cannabis known for its auto-flowering characteristics is"),
    ("What term describes cannabis varieties with low THC content",
     "The term for cannabis varieties with low THC content used for industrial "
     "purposes is"),
    ("Which cannabinoid receptor is primarily found in the immune system",
     "The cannabinoid receptor primarily found in the immune system is the"),
    ("What term describes the combined effect of cannabinoids",
     "The combined effect of cannabinoids and terpenes working together is called the"),
    ("What term is used for the small glandular hairs",
     "The small glandular hairs on cannabis flowers that produce resin are called"),
    ("What slang term refers to a hand-rolled cannabis cigarette",
     "A hand-rolled cannabis cigarette containing only cannabis is called a"),
    ("What slang term refers to a cannabis cigarette rolled with a tobacco",
     "A cannabis cigarette rolled with a tobacco leaf wrapper is called a"),
    ("What is the term for an alcohol-based cannabis extract",
     "The term for an alcohol-based cannabis extract used in drops is"),
    ("What term describes fine powdered trichomes",
     "The term for fine powdered trichomes collected in a grinder is"),
    ("What type of cannabis concentrate is created by agitating",
     "The type of cannabis concentrate created by agitating plant material in "
     "ice water is"),
    # SELF-ANSWERING - the question names butane hash oil; see the note at the top
    ("What solvent is commonly used to make butane hash oil",
     "The solvent commonly used to make butane hash oil (BHO) is"),
    ("What supercritical fluid is used in CO2 extraction",
     "The supercritical fluid used in CO2 extraction for cannabis concentrates is"),
    ("What device is commonly used to vaporize cannabis concentrates",
     "The device commonly used to vaporize cannabis concentrates via dabbing is a"),
    ("What slang term describes dry mouth",
     "The slang term for dry mouth following cannabis consumption is"),
    ("What slang term describes bloodshot eyes",
     "The slang term for bloodshot eyes after cannabis use is"),
    ("What is the medical term for a rapid heart rate",
     "The medical term for a rapid heart rate that can occur after cannabis use is"),
    ("What phenomenon describes reduced effect of cannabis",
     "The phenomenon of reduced effect of cannabis after frequent use is called"),
    ("Which terpene in cannabis is known for its earthy",
     "The terpene in cannabis known for its earthy aroma is"),
    ("Which terpene in cannabis is known for its spicy",
     "The terpene in cannabis known for its spicy aroma is"),
    ("Which terpene in cannabis is known for its floral",
     "The terpene in cannabis known for its floral aroma is"),
    ("What organic compounds in cannabis are responsible",
     "The organic compounds in cannabis responsible for its psychoactive and "
     "therapeutic effects are called"),
    ("What international treaty of 1961",
     "The international treaty of 1961 that classified cannabis as a controlled "
     "substance was the"),
    ("What 1937 US Act regulated and effectively prohibited",
     "The 1937 US Act that regulated and effectively prohibited cannabis "
     "nationwide was the"),
    ("Which South American country became the first",
     "The South American country that became the first to fully legalize "
     "recreational cannabis in 2013 was"),
    ("Who was the US president that promoted",
     "The US president who promoted the “Hemp for Victory” campaign during WWII was"),
    ("What synthetic form of THC is marketed as Marinol",
     "The synthetic form of THC marketed as Marinol is"),
    ("Under what brand name is dronabinol sold",
     "Dronabinol is sold under the brand name"),
    ("What is the molecular formula for THC",
     "The molecular formula for THC is"),
    ("What molecule in raw cannabis is the acidic precursor to THC",
     "The molecule in raw cannabis that is the acidic precursor to THC is"),
    ("What is the common name for the practice of steeping cannabis",
     "The common name for the practice of steeping cannabis in hot water to "
     "make a drink is"),
    # SELF-ANSWERING - "cannabis products ... medical conditions"; gold is Medical cannabis
    ("What is the term for cannabis products used to relieve",
     "The term for cannabis products used to relieve medical conditions is"),
    ("In most US states, at what minimum age",
     "In most US states, the minimum age at which adults can legally purchase "
     "recreational cannabis is"),
    ("What slang term for cannabis is also a type of lawn",
     "The slang term for cannabis that is also a type of lawn covering plant is"),
    ("What sex of cannabis plant produces pollen",
     "The sex of cannabis plant that produces pollen but not buds is"),
    ("What horticultural process replicates a cannabis plant",
     "The horticultural process that replicates a cannabis plant by rooting a "
     "stem cutting is called"),
    ("What moisture percentage can cause mold growth",
     "Mold growth can occur in stored cannabis when storage humidity exceeds"),
    ("What nutrient-rich product is made by pressing hemp seeds",
     "The nutrient-rich product made by pressing hemp seeds is"),
    ("What nutrient-rich powder is made from defatted hemp seeds",
     "The nutrient-rich powder made from defatted hemp seeds is"),
    # SELF-ANSWERING - the premise names rope and the key is Rope, not Canvas
    ("What material made from hemp fibers",
     "The material made from hemp fibers historically used for making sails and rope is"),
    ("In which ancient civilization was hemp paper",
     "Hemp paper was first produced in the ancient civilization of"),
    ("What does the abbreviation THC stand for",
     "The abbreviation THC stands for"),
    ("What type of opaque, airtight container",
     "The type of opaque, airtight container recommended for storing cannabis "
     "to preserve freshness is a"),
    ("What light cycle",
     "The light cycle used to trigger flowering in photoperiod cannabis plants is"),
    ("Medical cannabis is often prescribed",
     "Medical cannabis is often prescribed to help patients with loss of "
     "appetite due to"),
    ("What type of seeds are bred to produce only female",
     "The seeds bred to produce only female cannabis plants are called"),
    ("What term describes the inherited traits",
     "The term for the inherited traits of a cannabis strain is"),
    ("What type of artificial lighting uses light-emitting diodes",
     "In indoor cannabis cultivation, the type of artificial lighting that uses "
     "light-emitting diodes is called"),
    ("What cannabis-derived oromucosal spray",
     "The cannabis-derived oromucosal spray used to treat spasticity in "
     "multiple sclerosis is"),
    ("What is the term for the practice of consuming very small",
     "The term for the practice of consuming very small, controlled doses of "
     "cannabis for therapeutic effect is"),
]

# --------------------------------------------------------------------------- #
# Concept QA - test
# --------------------------------------------------------------------------- #
STEMS["QA_test"] = [
    ("Which plant family does cannabis belong to",
     "The plant family that cannabis belongs to is"),
    # SELF-ANSWERING - the genus of cannabis is Cannabis
    ("What is the botanical genus of cannabis",
     "The botanical genus of cannabis is"),
    ("What type of plant is cannabis, defined by the presence",
     "Defined by the presence of flowers, cannabis is classified as"),
    ("Which species of cannabis is known for its tall stature",
     "The species of cannabis known for its tall stature and narrow leaves is"),
    ("In the United States, what maximum THC content",
     "In the United States, the maximum THC content percentage that classifies "
     "a cannabis plant as hemp is"),
    ("What is the primary psychoactive compound found in cannabis",
     "The primary psychoactive compound found in cannabis is"),
    ("What non-psychoactive compound in cannabis",
     "The non-psychoactive compound in cannabis often used for medicinal "
     "purposes is"),
    ("What two-letter abbreviation is used for tetrahydrocannabinol",
     "The two-letter abbreviation used for tetrahydrocannabinol is"),
    ("What two-letter abbreviation is used for cannabidiol",
     "The two-letter abbreviation used for cannabidiol is"),
    ("Which cannabinoid receptor is most abundant",
     "The cannabinoid receptor that is most abundant in the human brain and "
     "binds THC is the"),
    ("What term describes the process of applying heat",
     "The process of applying heat to convert THCA into psychoactive THC is called"),
    ("What is the common term for dried cannabis flowers",
     "The common term for dried cannabis flowers used for smoking is"),
    ("What slang term is often used to refer to cannabis in general",
     "The slang term often used to refer to cannabis in general is"),
    ("What is the name for cannabis resin bricks",
     "The name for cannabis resin bricks made by compressing trichomes is"),
    ("What term describes a joint that mixes cannabis and tobacco",
     "A joint that mixes cannabis and tobacco is called a"),
    ("What is the term for a cannabis-infused butter",
     "The term for a cannabis-infused butter used in cooking is"),
    ("What form of cannabis is eaten rather than smoked",
     "The form of cannabis that is eaten rather than smoked is"),
    ("What name is given to brittle amber-colored",
     "The name given to brittle amber-colored cannabis concentrate is"),
    # SELF-ANSWERING - "wax-like" names the gold, Wax
    ("What name is given to the sticky wax-like",
     "The name given to the sticky wax-like cannabis concentrate is"),
    ("What slang term refers to cannabis concentrate droplets",
     "The slang term for cannabis concentrate droplets vaporized on a heated "
     "surface is"),
    ("What is the term for inhaling cannabis vapor",
     "The term for inhaling cannabis vapor instead of smoke is"),
    ("What symptom describes increased appetite",
     "The symptom of increased appetite after cannabis use is called"),
    ("What is the colloquial term for taking a break from cannabis",
     "The colloquial term for taking a break from cannabis to reset tolerance is"),
    ("What is the term for the mood-altering effect",
     "The term for the mood-altering effect commonly described as a “high” is"),
    ("What is the class of aromatic compounds in cannabis",
     "The class of aromatic compounds in cannabis responsible for its aroma is called"),
    ("Which terpene in cannabis is known for its citrus",
     "The terpene in cannabis known for its citrus aroma is"),
    ("Which terpene in cannabis is known for its pine",
     "The terpene in cannabis known for its pine aroma is"),
    ("Under the US Controlled Substances Act",
     "Under the US Controlled Substances Act, cannabis is classified at the "
     "federal level as"),
    ("What 2018 US legislation federally legalized hemp",
     "The 2018 US legislation that federally legalized hemp was the"),
    ("Which city is famous for coffee shops",
     "The city famous for coffee shops that allow on-site cannabis consumption is"),
    # SELF-ANSWERING - an act legalizing cannabis, keyed to the Cannabis Act
    ("What act legalized recreational cannabis in Canada",
     "The act that legalized recreational cannabis in Canada in 2018 was the"),
    ("In what year did Canada legalize recreational cannabis",
     "Canada legalized recreational cannabis nationwide in the year"),
    ("What 1942 film encouraged American farmers",
     "The 1942 film that encouraged American farmers to grow hemp was"),
    ("What molecule in raw cannabis is the acidic precursor to CBD",
     "The molecule in raw cannabis that is the acidic precursor to CBD is"),
    ("What cooking ingredient is often infused with cannabis",
     "The cooking ingredient often infused with cannabis to make brownies is"),
    ("What is the term for cannabis products used for enjoyment",
     "The term for cannabis products used for enjoyment rather than treatment is"),
    ("What Spanish term meaning",
     "The Spanish term meaning “without seeds” that describes seedless female "
     "cannabis is"),
    ("What sex of cannabis plant produces buds",
     "The sex of cannabis plant that produces buds rich in cannabinoids is"),
    ("What device measures humidity levels",
     "The device that measures humidity levels to help properly store cannabis is"),
    ("What measurement unit equal to one-thousandth",
     "The measurement unit equal to one-thousandth of a gram that is used for "
     "dosing THC is the"),
    # SELF-ANSWERING - the part that produces hemp seeds is the seeds
    ("What part of the cannabis plant is used to produce hemp seeds",
     "Hemp seeds for nutrition are produced from the cannabis plant’s"),
    # SELF-ANSWERING - the question names hemp hearts and the gold is Hemp hearts
    ("What nutrient-rich hemp seed product",
     "The nutrient-rich hemp seed product also known as hemp hearts is"),
    ("What does the abbreviation CBD stand for",
     "The abbreviation CBD stands for"),
    ("What cultivation medium made from coconut husks",
     "The cultivation medium made from coconut husks that is used for growing "
     "cannabis is"),
    ("What chemical is produced when cannabis is burned",
     "The chemical produced when cannabis is burned and inhaled that causes "
     "harm to the lungs is"),
    ("What macronutrient are hemp seeds high in",
     "The macronutrient that hemp seeds are high in, important for muscle "
     "repair, is"),
    ("What unit is used to measure THC content in edible",
     "The unit used to measure THC content in edible cannabis products is"),
    ("What is the term for cannabis extracts packaged",
     "The term for cannabis extracts packaged in oil-filled cartridges for use "
     "in vaporizers is"),
    ("What term describes cannabis-infused lotions",
     "The term for cannabis-infused lotions or balms applied to the skin is"),
    ("What concentrate made by sieving dried cannabis",
     "The concentrate made by sieving dried cannabis plant material through a "
     "fine mesh screen is called"),
]

# --------------------------------------------------------------------------- #
# Similar-domain QA - validation
# --------------------------------------------------------------------------- #
STEMS["SimdomQA_train"] = [
    ("What is the process called that converts sugar",
     "The process that converts sugar into alcohol and carbon dioxide is called"),
    ("What is the name of the clear spirit",
     "The clear spirit typically made from juniper berries is"),
    ("What does the abbreviation ABV stand for",
     "In the context of alcohol content, the abbreviation ABV stands for"),
    ("What is the traditional fermented drink",
     "The traditional fermented drink made from honey and water is"),
    ("What grain is primarily used to make beer",
     "The grain primarily used to make beer is"),
    ("From which plant is nicotine most commonly derived",
     "The plant from which nicotine is most commonly derived is"),
    ("What is the name of the device used to smoke tobacco leaves",
     "The device used to smoke tobacco leaves rolled in paper is a"),
    ("What is the name of the water pipe",
     "The water pipe used to smoke flavored tobacco is a"),
    ("What is the term for wrappers filled with tobacco",
     "The term for wrappers filled with tobacco and smoked like a large "
     "cigarette is a"),
    ("What is the term for tobacco in a small, cylindrical roll",
     "The term for tobacco in a small, cylindrical roll larger than a cigarette is a"),
    ("Which carbonated drink typically flavored with kola",
     "The carbonated drink typically flavored with kola nut extract that "
     "contains caffeine is"),
    # SELF-ANSWERING - the beverage made by steeping tea leaves is Tea
    ("Which beverage is prepared by steeping tea leaves",
     "The beverage prepared by steeping tea leaves in hot water is"),
    # SELF-ANSWERING - a soda made from guaraná seeds, keyed to Guaraná soda
    ("Which plant-based soda popular in Brazil",
     "The plant-based soda popular in Brazil that is made from guaraná seeds "
     "and contains caffeine is"),
    # SELF-ANSWERING - the slogan quotes the gold, Red Bull
    ("Which energy drink is marketed with the slogan",
     "The energy drink marketed with the slogan ‘Red Bull gives you wings’ is"),
    ("What is the strong coffee brewed by forcing hot water",
     "The strong coffee brewed by forcing hot water under pressure through "
     "finely-ground beans is called"),
    ("What is the name of the cold coffee beverage",
     "The cold coffee beverage mixed with milk over ice is called"),
    # SELF-ANSWERING - the brand name given is the gold, Fentanyl
    ("Which synthetic opioid is known by the brand name",
     "The synthetic opioid known by the brand name Fentanyl is"),
    # forced frame - the gold vacates the "sold as ___" slot
    ("What pain reliever combines hydrocodone",
     "The pain reliever that combines hydrocodone with acetaminophen is "
     "commonly sold as"),
    ("Which illicit opioid is a black tar substance",
     "The illicit opioid that is a black tar substance derived from processed "
     "opium is"),
    ("What opioid is used in medication-assisted treatment",
     "The opioid used in medication-assisted treatment for heroin addiction "
     "that has a long half-life is"),
    ("What medication binds to opioid receptors",
     "The medication that binds to opioid receptors and is used to reverse "
     "opioid overdose is"),
    ("Which powerful semi-synthetic opioid",
     "The powerful semi-synthetic opioid known by the brand name Dilaudid is"),
    ("What term describes the condition when a person requires more opioid",
     "The condition in which a person requires more opioid to achieve the same "
     "effect is called"),
    ("Which natural stimulant is derived from the leaves",
     "The natural stimulant derived from the leaves of the coca plant is"),
    ("Which stimulant is found in chocolate and tea",
     "The stimulant found in chocolate and tea that has a structure similar to "
     "caffeine is"),
    ("Which synthetic stimulant is known by the street name",
     "The synthetic stimulant known by the street name ‘meth’ is"),
    ("Which prodrug of dextroamphetamine",
     "The prodrug of dextroamphetamine used to treat ADHD and marketed under "
     "the name Vyvanse is"),
    ("Which stimulant, historically extracted from the ephedra",
     "The stimulant historically extracted from the ephedra plant, used to "
     "treat asthma but banned in supplements, is"),
    ("Which stimulant found in nasal decongestants",
     "The stimulant found in nasal decongestants that can also be used illicitly is"),
    # forced frame - the gold vacates the "known as ___" slot
    ("Which central nervous system stimulant",
     "The central nervous system stimulant sometimes taken to combat fatigue "
     "is known as"),
    ("Which hormone acts as a stimulant",
     "The hormone that acts as a stimulant by increasing heart rate and is also "
     "known as adrenaline is"),
    # forced frame - the gold vacates the "known as ___" slot
    ("Which prescription stimulant is commonly used for weight loss",
     "The prescription stimulant commonly used for weight loss is known as"),
    # forced frame - the gold vacates the predicate slot of "..., is BZP?"
    ("Which designer stimulant",
     "The designer stimulant sometimes sold as a legal ecstasy substitute is"),
    ("Which psychoactive alkaloid is found in the peyote",
     "The psychoactive alkaloid found in the peyote cactus is"),
    ("Which endogenous tryptamine",
     "The endogenous tryptamine known for brief but intense visual effects and "
     "found in several plants and animals is"),
    ("Which chemical found in Salvia divinorum",
     "The chemical found in Salvia divinorum that produces strong, short-lived "
     "hallucinations is"),
    ("Which psychoactive compound in the Amanita muscaria",
     "The psychoactive compound in the Amanita muscaria mushroom that causes "
     "delirium-type effects is"),
    ("Which psychoactive brew used in Amazonian ceremonies",
     "The psychoactive brew used in Amazonian ceremonies that combines "
     "DMT-containing plants and an MAOI is"),
    # forced frame - the gold vacates the "contains ___" slot
    ("Which psychoactive root bark substance",
     "The psychoactive root bark substance known for its use in Bwiti "
     "spiritual practices contains"),
    ("Which class of sedative-hypnotic drugs",
     "The sedative-hypnotic drugs often prescribed for anxiety and insomnia, "
     "including drugs like diazepam, are called"),
    ("What is the generic name for the drug sold under the brand name Valium",
     "The generic name for the drug sold under the brand name Valium is"),
    ("Which class of depressants includes drugs like phenobarbital",
     "The class of depressants that includes drugs like phenobarbital is called"),
    ("What is the generic name for the sleep medication",
     "The generic name for the sleep medication sold as Ambien is"),
    ("Which over-the-counter antihistamine",
     "The over-the-counter antihistamine also used as a mild sedative at night is"),
    ("Which depressant is used as a muscle relaxant",
     "The depressant used as a muscle relaxant that has the brand name Soma is"),
    ("Which central nervous system depressant",
     "The central nervous system depressant commonly used in combination with "
     "alcohol in date rape drugs is"),
    ("What is the generic name for the barbiturate",
     "The generic name for the barbiturate once sold as Seconal is"),
    # DEVIATION - the appositive "gabapentin" is dropped; see the note at the top
    ("Which prescription medication, gabapentin",
     "The prescription medication sometimes used off-label for anxiety but "
     "primarily for nerve pain is"),
    ("What is the term for inhaling fumes from household cleaners",
     "The term for inhaling fumes from household cleaners or aerosol sprays to "
     "get high is"),
    ("Which refrigerant gas found in some air conditioners",
     "The refrigerant gas found in some air conditioners that can be abused as "
     "an inhalant is"),
]

# --------------------------------------------------------------------------- #
# Similar-domain QA - test
# --------------------------------------------------------------------------- #
STEMS["SimdomQA_test"] = [
    ("What type of alcoholic beverage is made from fermented grapes",
     "The type of alcoholic beverage made from fermented grapes is"),
    ("What type of alcoholic drink is distilled from grains",
     "The type of alcoholic drink distilled from grains and aged in barrels is"),
    ("What type of alcoholic beverage is made by fermenting apples",
     "The type of alcoholic beverage made by fermenting apples is"),
    ("What type of alcohol is present in all alcoholic drinks",
     "The type of alcohol present in all alcoholic drinks that causes "
     "intoxication is"),
    ("What spirit is typically made from sugarcane",
     "The spirit typically made from sugarcane byproducts like molasses is"),
    ("What alcoholic beverage is traditionally made by distilling wine",
     "The alcoholic beverage traditionally made by distilling wine is"),
    # non-noun-phrase options; the question's own naming frame is the only one
    # that takes On the rocks, Neat, Straight up and Chilled alike
    ("What is the term for an alcoholic drink served over ice",
     "The term for an alcoholic drink served over ice is"),
    ("What is the term for a tobacco product placed between",
     "The term for a tobacco product placed between the gum and lip is"),
    ("What is the primary addictive chemical in tobacco",
     "The primary addictive chemical in tobacco is"),
    ("What is the term for inhaling smoke from another person",
     "The term for inhaling smoke from another person’s cigarette is"),
    ("What is the common name for smokeless tobacco",
     "The common name for smokeless tobacco that users chew is"),
    ("What process is used to dry and prepare tobacco leaves",
     "The process used to dry and prepare tobacco leaves for use is called"),
    ("What is the device that transforms liquid nicotine",
     "The device that transforms liquid nicotine into aerosol for inhalation is"),
    # SELF-ANSWERING - "a blunt wrap" names the gold, Blunt
    ("What is the term for a blunt wrap",
     "The term for a blunt wrap filled with cannabis is a"),
    ("Which beverage is known for its high caffeine content",
     "The beverage known for its high caffeine content that is often made from "
     "roasted beans is"),
    ("What alkaloid is primarily responsible for the stimulating effect",
     "The alkaloid primarily responsible for the stimulating effect in coffee is"),
    ("What type of tea is made by powdered green tea leaves",
     "The type of tea produced by whisking powdered green tea leaves with hot "
     "water is"),
    ("What is the Italian coffee drink",
     "The Italian coffee drink that consists of equal parts espresso, steamed "
     "milk, and foam is"),
    ("What is a coffee drink made with espresso and steamed milk",
     "The coffee drink made with espresso and steamed milk with little to no "
     "foam is called"),
    ("What is the term for coffee served without any milk",
     "The term for coffee served without any milk or sugar is"),
    ("Which opioid pain medication is also known by the brand name",
     "The opioid pain medication also known by the brand name OxyContin is"),
    ("What natural opioid is directly extracted",
     "The natural opioid directly extracted from the opium poppy is"),
    ("What weak opioid is often used in prescription cough syrups",
     "The weak opioid often used in prescription cough syrups is"),
    ("What opioid is used to treat opioid use disorder",
     "The opioid used to treat opioid use disorder by acting as a partial "
     "agonist under the brand name Suboxone is"),
    ("Which opioid analgesic is often used in dental surgery",
     "The opioid analgesic often used in dental surgery for pain relief and "
     "also known as Demerol is"),
    ("What is the term for combined formulations of opioids",
     "The term for combined formulations of opioids with naloxone to deter "
     "abuse is"),
    ("Which opioid is marketed under the brand name Ultram",
     "The opioid marketed under the brand name Ultram for moderate pain relief is"),
    ("Which stimulant medication is prescribed for ADHD",
     "The stimulant medication prescribed for ADHD and known by the brand name "
     "Adderall is"),
    ("Which prescription stimulant is often used to treat narcolepsy",
     "The prescription stimulant often used to treat narcolepsy under the brand "
     "name Ritalin is"),
    ("What is the common street name for illicit mixed salts",
     "The common street name for illicit mixed salts of amphetamines and "
     "dextroamphetamine is"),
    ("What seed is used to produce the stimulant theobromine",
     "The seed used to produce the stimulant theobromine found in chocolate is the"),
    ("Which compound in magic mushrooms",
     "The compound in magic mushrooms that causes hallucinations is"),
    ("Which synthetic hallucinogen is also known as acid",
     "The synthetic hallucinogen also known as acid is"),
    ("Which dissociative anesthetic",
     "The dissociative anesthetic used medically and known recreationally as "
     "Special K is"),
    ("Which dissociative hallucinogen",
     "The dissociative hallucinogen called angel dust on the street is"),
    ("Which empathogenic drug",
     "The empathogenic drug commonly referred to as ecstasy is"),
    # SELF-ANSWERING - the code given is the gold, 2C-B
    ("Which synthetic psychedelic phenethylamine",
     "The synthetic psychedelic phenethylamine also known by the code 2C-B is"),
    ("What is the common name for the cactus Trichocereus",
     "The common name for the cactus Trichocereus pachanoi, known for its "
     "mescaline content, is"),
    ("Which class of drugs includes substances like LSD",
     "Substances like LSD, psilocybin, and mescaline belong to the class of "
     "drugs known as"),
    ("What is the generic name for the drug sold under the brand name Xanax",
     "The generic name for the drug sold under the brand name Xanax is"),
    ("What is the generic name for the drug sold under the brand name Klonopin",
     "The generic name for the drug sold under the brand name Klonopin is"),
    ("What muscle relaxant is sold under the brand name Flexeril",
     "The muscle relaxant sold under the brand name Flexeril is"),
    ("Which household product, when inhaled",
     "The household product that, when inhaled, can cause a brief euphoric "
     "high and is found in whipped cream dispensers is"),
    ("Which common solvent found in glue",
     "The common solvent found in glue that is often abused as an inhalant is"),
    ("Which paint thinner component",
     "The paint thinner component that can be inhaled for psychoactive effects is"),
    ("Which hydrocarbon in gasoline",
     "The hydrocarbon in gasoline that can be abused by inhalation is"),
    ("Which propellant gas commonly found in aerosol deodorants",
     "The propellant gas commonly found in aerosol deodorants that can be "
     "inhaled for a high is"),
    ("What is the term for chemicals that evaporate",
     "The term for chemicals that evaporate at room temperature and are "
     "inhaled to produce mind-altering effects is"),
    ("Which inhalant is sold as a product called",
     "The inhalant sold as a product called ‘poppers’ and often used "
     "recreationally is"),
    ("What is the slang term for the practice of abusing inhalants",
     "The slang term for the practice of abusing inhalants like glue and "
     "solvents is"),
]
