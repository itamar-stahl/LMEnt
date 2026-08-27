#!/usr/bin/env python
"""Declarative-stem rewrites of EMBER's World War II questions.

Same rules as stems_pornography.py, which is the reference copy of them:

  1. Declarative prose, no question mark, no "Question:".
  2. All four options continue the stem grammatically.
  3. Written from the question, never from the answer key.
  4. Stop before a determiner unless all four options take it.
  5. Options, answer key, file order and shuffle seed unchanged.

This concept is the easiest of the four for rule 2 and the hardest for rule 4.
Almost every option is a proper noun, a date or a place, so a plain "...was"
frame takes all four without strain. The articles are what bite. "...was the"
is safe over Treaty of Versailles / Saint-Germain / Trianon / Brest-Litovsk and
over four Battle-of-X names, and unsafe wherever an Operation sits next to a
Battle: "the Operation Overlord" is not English, so SimdomQA_test #39 stops at
"was called". Same trap in QA_train #34 (Dieppe Raid against Operation
Jubilee), QA_test #47 (Zhejiang-Jiangxi campaign against Operation Ichi-Go) and
SimdomQA_train #27 (Bay of Pigs Invasion against Operation Mongoose).

The mirror-image trap is options that carry their own "The". Four sets are all
"The X" (The Eastern Front and its three fronts; The Commissar Order and its
three decrees; The tank / The jeep / The armored car / The submarine; The North
/ The South / The West / The Border States), so those stems stop at "was" and
let the option supply the article.

One deviation from the original wording, following the precedent set by the
Shunga item in stems_pornography.py:

  QA_train "What port in western Egypt did Axis forces besiege in early 1941?"
      -> the stem drops **in western Egypt**. The gold answer is Tobruk, which
      is in Libya, while two of the three distractors (Alexandria, Cairo) are
      in Egypt. The premise therefore contradicts its own answer key and points
      at the distractors, so a declarative stem carrying it would push the
      model away from the gold option. Note the contrast with QA_train #38,
      "At what Egyptian location were Axis forces stopped in North Africa?",
      where the gold answer El Alamein *is* Egyptian: that premise is accurate
      and is kept, hint and all.

Everything else keeps the original's content words.
"""
from __future__ import annotations

STEMS: dict[str, list[tuple[str, str]]] = {}

# --------------------------------------------------------------------------- #
# Concept QA - validation
# --------------------------------------------------------------------------- #
STEMS["QA_train"] = [
    ("Who led the Free French Forces",
     "The Free French Forces that assisted in the liberation of Paris were led by"),
    ("What country did the Soviet Union sign a non-aggression pact",
     "On 23 August 1939, the Soviet Union signed a non-aggression pact with"),
    ("What client state did Mussolini establish",
     "After being rescued by German special forces, Mussolini established the "
     "client state called the"),
    ("At what river did American and Soviet forces meet",
     "On 25 April 1945, American and Soviet forces met at the"),
    ("What port in western Egypt did Axis forces besiege",
     "In early 1941, Axis forces besieged the port of"),
    ("What type of weapon was delivered for the only time",
     "The type of weapon delivered for the only time in war during World War II was"),
    ("Who replaced Neville Chamberlain",
     "On 10 May 1940, Neville Chamberlain was replaced as British Prime Minister by"),
    ("Who succeeded Franklin D. Roosevelt",
     "Franklin D. Roosevelt was succeeded as President of the United States by"),
    ("What Chinese capital did Japan capture",
     "In December 1937, Japan captured the Chinese capital of"),
    ("What treaty imposed significant territorial and financial losses",
     "The treaty that imposed significant territorial and financial losses on "
     "Germany after WWI was the"),
    ("Which country invaded Poland on 1 September 1939",
     "The country that invaded Poland on 1 September 1939 was"),
    ("The fall of which city to Soviet troops",
     "The city whose fall to Soviet troops culminated the war in Europe was"),
    ("Which country invaded Manchuria in 1931",
     "The country that invaded Manchuria in 1931 was"),
    ("In 1944, where did the Western Allies invade France",
     "In 1944, the Western Allies invaded France at"),
    ("When did World War II end",
     "World War II ended on"),
    ("In what battle did the US decisively defeat Japanese forces",
     "In mid-June 1944, the US decisively defeated Japanese forces at the"),
    ("What strong fortifications on the Franco-German border",
     "Germany circumvented the strong fortifications on the Franco-German "
     "border known as the"),
    ("In what battle did an Allied task force fight Japanese naval forces",
     "An Allied task force fought Japanese naval forces to a draw and thwarted "
     "the invasion of Port Moresby at the"),
    ("What name was given to the Soviet forced labour camps",
     "The name given to the Soviet forced labour camps where deported civilians "
     "and POWs were imprisoned was"),
    ("What march resulted in thousands of deaths",
     "The march that resulted in thousands of deaths of Filipino and American "
     "prisoners in 1942 was"),
    ("Who led the Nationalist rebels",
     "In the Spanish Civil War, the Nationalist rebels were led by"),
    ("What territory did Italy desire as a colonial possession",
     "The territory Italy desired as a colonial possession, leading to a war in "
     "1935, was"),
    ("Which two countries declared war on Germany",
     "The two countries that declared war on Germany two days after the "
     "invasion of Poland were"),
    ("In what region did Germany launch its last massive counter-offensive",
     "In December 1944, Germany launched its last massive counter-offensive on "
     "the Western Front in"),
    ("At what location did the first German attack",
     "The first German attack against Polish defences occurred at"),
    ("What naval strategy did the Western Allies use",
     "The naval strategy the Western Allies used to damage Germany's economy was"),
    ("What region did Hitler remilitarise",
     "In March 1936, Hitler remilitarised"),
    ("What title did Hitler proclaim for himself",
     "After President Hindenburg's death in 1934, the title Hitler proclaimed "
     "for himself was"),
    ("What uprising in Poland was quelled",
     "The uprising in Poland that was quelled by the Germans, resulting in over "
     "150,000 Polish deaths, was the"),
    ("What major Soviet city did the German army almost reach",
     "Before being halted by harsh weather and fierce battles, the German army "
     "almost reached the major Soviet city of"),
    ("What front did Germany open in June 1941",
     "By invading the Soviet Union in June 1941, Germany opened"),
    ("What is the estimated death toll",
     "The estimated death toll of World War II is"),
    ("On what date did the Second Sino-Japanese War start",
     "The Second Sino-Japanese War started on"),
    ("What failed raid demonstrated",
     "The failed raid that demonstrated the Western Allies' inability to launch "
     "a continental invasion without better preparation was"),
    ("What pact formally united Japan, Italy, and Germany",
     "In September 1940, Japan, Italy, and Germany were formally united as the "
     "Axis powers by the"),
    ("Who was the leader of Nazi Germany during the invasion of Poland",
     "During the invasion of Poland, the leader of Nazi Germany was"),
    ("What puppet state did Japan establish",
     "After invading Manchuria, Japan established the puppet state of"),
    ("At what Egyptian location were Axis forces stopped",
     "In North Africa, Axis forces were stopped at the Egyptian location of"),
    ("Who led the fascist movement that seized power in Italy",
     "The fascist movement that seized power in Italy from 1922 to 1925 was led by"),
    ("What German order entailed executing",
     "The German order that entailed executing all Jewish and Communist POWs "
     "immediately was"),
    ("In what city did Roosevelt, Churchill, and Chiang Kai-shek meet",
     "In November 1943, Roosevelt, Churchill, and Chiang Kai-shek met in the city of"),
    ("Which two nations emerged as rival superpowers",
     "The two nations that emerged as rival superpowers after WWII were"),
    ("What declaration did the Allied Big Four",
     "On 1 January 1942, the Allied Big Four and 22 smaller governments issued the"),
    ("What two political entities controlled the western and eastern occupation zones",
     "The western and eastern occupation zones of Germany and Austria were "
     "controlled by"),
    ("In June 1940, which major Western European country",
     "In June 1940, the major Western European country that fell to Germany was"),
    ("What organization was established in 1920",
     "The organization established in 1920 by the Paris Peace Conference to "
     "prevent future wars was the"),
    ("Which country sustained the highest overall death toll",
     "The country that sustained the highest overall death toll in World War II was"),
    ("Which two rival international military alliances",
     "The two rival international military alliances that formalised the "
     "post-war division of the world were"),
    ("What country experienced a pro-British nationalist overthrow",
     "The country that experienced a pro-British nationalist overthrow in late "
     "March 1941 before being invaded by Germany was"),
    ("What island became a focal point",
     "Starting in July 1942, the island that became a focal point for a major "
     "campaign in the southern Solomon Islands was"),
]

# --------------------------------------------------------------------------- #
# Concept QA - test
# --------------------------------------------------------------------------- #
STEMS["QA_test"] = [
    ("In what June 1942 battle were Japanese advances",
     "Japanese advances in the Pacific were halted in June 1942 at the"),
    ("When did the Soviet Union invade Poland",
     "The Soviet Union invaded Poland on"),
    ("What major Soviet city had its lethal siege ended",
     "The major Soviet city whose lethal siege ended in January 1944 was"),
    ("What country invaded Finland",
     "The country that invaded Finland in November 1939, leading to the Winter "
     "War, was"),
    ("What US fleet did Japan aim to neutralize",
     "At Pearl Harbor, Japan aimed to neutralize the US fleet known as the"),
    ("What area of Czechoslovakia did Hitler claim",
     "Due to its ethnic German population, the area of Czechoslovakia Hitler "
     "claimed was"),
    ("What agreement created the International Monetary Fund",
     "The International Monetary Fund and the International Bank for "
     "Reconstruction and Development were created by the"),
    ("What major tank battle occurred around a bulge",
     "In July 1943, the major tank battle that occurred around a bulge in the "
     "Soviet lines was the"),
    ("What country did Germany annex",
     "In March 1938, Germany annexed"),
    ("What massive scorched earth initiative",
     "In North China, the massive scorched earth initiative implemented by "
     "Japanese armies was"),
    ("What organization was created after the war",
     "The organization created after the war to foster international "
     "cooperation was"),
    ("Along what parallel was Korea divided",
     "After the war, Korea was divided along the"),
    ("On what date did Germany unconditionally surrender",
     "Germany unconditionally surrendered on"),
    ("On what date did the Western Allies invade northern France",
     "The Western Allies invaded northern France (D-Day) on"),
    ("Through what region did the Germans carry out a flanking manoeuvre",
     "The Germans carried out a flanking manoeuvre into France through"),
    ("When did World War II officially begin",
     "World War II officially began on"),
    ("What axis did Germany and Italy form",
     "In October 1936, Germany and Italy formed the"),
    ("What was the unoccupied rump state",
     "The unoccupied rump state in southern France was called"),
    ("What was the aerial battle fought between Britain and Germany",
     "The aerial battle fought between Britain and Germany was called"),
    ("What territory did Hitler legally reunite",
     "In early 1935, the territory Hitler legally reunited with Germany was"),
    ("What were the two opposing coalitions",
     "The two opposing coalitions in World War II were called"),
    ("What country did Italy conquer",
     "In April 1939, Italy conquered"),
    ("What international organization officially came into existence",
     "The international organization that officially came into existence on "
     "24 October 1945 was the"),
    ("What two countries did Germany invade in April 1940",
     "In April 1940, to protect iron ore shipments, Germany invaded"),
    ("Where were Axis forces defeated in the Soviet Union",
     "In early 1943, Axis forces in the Soviet Union were defeated at"),
    ("What proxy war did Germany and the Soviet Union use",
     "The proxy war Germany and the Soviet Union used to test weapons and "
     "tactics was"),
    ("What British possession did the Italian Regia Aeronautica besiege",
     "In early June 1940, the British possession besieged by the Italian Regia "
     "Aeronautica was"),
    ("What pact partitioned Poland",
     "Poland was partitioned between Germany and the Soviet Union by the"),
    ("What democratic government was created in Germany",
     "The democratic government created in Germany after WWI was"),
    ("On what date did Germany invade the Soviet Union",
     "Germany invaded the Soviet Union on"),
    ("What Japanese emperor approved the operations plan",
     "On 5 November 1941, the operations plan for the war was approved by"),
    ("Who became Prime Minister of Japan",
     "Before the Pearl Harbor attack, Fumimaro Konoe was replaced as Prime "
     "Minister of Japan by"),
    ("Who led the communist Partisans",
     "The communist Partisans in Yugoslavia were led by"),
    ("What pact did Germany and Japan sign",
     "In November 1936, Germany and Japan signed the"),
    ("What US territory in Hawaii did Japan attack",
     "In December 1941, the US territory in Hawaii that Japan attacked was"),
    ("On what battleship were the Japanese surrender documents signed",
     "The Japanese surrender documents were signed on board the"),
    ("What was the codename for Germany",
     "The codename for Germany's invasion of the Soviet Union was"),
    ("What country assisted Germany after the fall of France",
     "After the fall of France in June 1940, Germany was assisted by"),
    ("On what two Japanese cities did the US drop atomic bombs",
     "The US dropped atomic bombs on the two Japanese cities of"),
    ("Which German industrial centre suffered a major firebombing",
     "The German industrial centre that suffered a major firebombing by the "
     "British and Americans in 1943 was"),
    ("What alliance did Germany and Italy formalise",
     "Shortly after the Franco-British pledge to Poland, Germany and Italy "
     "formalised the"),
    ("How many Jews did the Nazis kill",
     "The number of Jews the Nazis killed in the Holocaust was"),
    ("Who commanded the German Afrika Korps",
     "The German Afrika Korps deployed to North Africa was commanded by"),
    ("On what date did Paris fall",
     "Paris fell to the Germans on"),
    ("What agreement conceded the Sudetenland",
     "The Sudetenland was conceded to Germany by the"),
    ("Where did Soviet, British, and US leaders meet",
     "In February 1945, Soviet, British, and US leaders met at the"),
    ("What Japanese campaign in China aimed to inflict retribution",
     "The Japanese campaign in China that aimed to inflict retribution for the "
     "Doolittle Raid was"),
    ("What Soviet city on the Volga River",
     "The Soviet city on the Volga River that became the site of a major battle "
     "where the German Sixth Army was encircled was"),
    ("What German battleship was sunk",
     "On 27 May 1941, the German battleship sunk by the British Home Fleet was the"),
    ("Which neutral country was jointly invaded",
     "In late August 1941, the neutral country jointly invaded by the British "
     "and Soviets to secure the Persian Corridor was"),
]

# --------------------------------------------------------------------------- #
# Similar-domain QA - validation
# --------------------------------------------------------------------------- #
STEMS["SimdomQA_train"] = [
    ("What was the alliance of Germany, Austria-Hungary",
     "The alliance of Germany, Austria-Hungary, and the Ottoman Empire was called"),
    ("Who was the famous nurse",
     "The famous nurse who founded the American Red Cross after the Civil War was"),
    ("Which major Southern city did General Sherman burn",
     "The major Southern city General Sherman burned on his march was"),
    ("Who was the President of the Confederate States",
     "The President of the Confederate States of America was"),
    ("What country did Germany invade first",
     "To start World War I, the first country Germany invaded was"),
    ("What ironclad ship fought for the Union",
     "The ironclad ship that fought for the Union was the"),
    ("What Asian country was divided at the 38th parallel",
     "The Asian country divided at the 38th parallel was"),
    ("What rival alliance was formed by the Soviet Union",
     "The rival alliance formed by the Soviet Union and Eastern Bloc countries was"),
    ("Which Union general led a destructive",
     "The Union general who led a destructive 'March to the Sea' was"),
    ("What Soviet leader introduced policies",
     "The Soviet leader who introduced policies of glasnost and perestroika was"),
    ("What was the name of the German submarines",
     "The German submarines used in World War I were known as"),
    ("Who was the first human to travel into space",
     "The first human to travel into space was"),
    ("What communist country famously split",
     "The communist country that famously split ideologically from the Soviet "
     "Union in the 1960s was"),
    ("Which Union general ultimately defeated Robert E. Lee",
     "The Union general who ultimately defeated Robert E. Lee was"),
    ("Who was the US President during World War I",
     "The US President during World War I was"),
    ("What empire was Franz Ferdinand heir to",
     "Franz Ferdinand was heir to the"),
    ("What battle was the longest of World War I",
     "The longest battle of World War I, lasting over 300 days, was the"),
    ("Which state was the first to secede",
     "The first state to secede from the Union was"),
    ("What year did the American Civil War begin",
     "The American Civil War began in the year"),
    ("What bloodiest single-day battle",
     "The bloodiest single-day battle in American history, fought in Maryland, "
     "was the"),
    ("What was the nickname given to American soldiers",
     "The nickname given to American soldiers in World War I was"),
    ("What country did the Soviet Union invade in 1979",
     "In 1979, the Soviet Union invaded"),
    ("Which Asian nation fought on the side of the Allies",
     "The Asian nation that fought on the side of the Allies in World War I was"),
    ("What term described the ideological divide",
     "The term that described the ideological divide between East and West in "
     "Europe was the"),
    ("What was the nickname for the Southern soldiers",
     "The nickname for the Southern soldiers was"),
    ("In what country was the Battle of the Somme fought",
     "The Battle of the Somme was fought in"),
    ("What failed US-backed invasion of Cuba",
     "The failed US-backed invasion of Cuba in 1961 was"),
    ("What document issued by Lincoln",
     "The document issued by Lincoln that freed slaves in Confederate states was"),
    ("Whose assassination triggered World War I",
     "World War I was triggered by the assassination of"),
    ("What was the capital of the Confederacy",
     "For most of the war, the capital of the Confederacy was"),
    ("What country intercepted the Zimmermann Telegram",
     "The Zimmermann Telegram was intercepted by"),
    ("Who was the first human to walk on the moon",
     "The first human to walk on the moon was"),
    ("Who was the communist leader of North Vietnam",
     "The communist leader of North Vietnam was"),
    ("What aircraft was shot down over the Soviet Union",
     "The aircraft shot down over the Soviet Union in 1960, causing a major "
     "incident, was the"),
    ("What European country remained neutral",
     "The European country that remained neutral but was surrounded by fighting "
     "nations was"),
    ("On what day is Armistice Day celebrated",
     "Armistice Day, marking the end of World War I fighting, is celebrated on"),
    ("Which major power exited World War I early",
     "The major power that exited World War I early due to a revolution in 1917 was"),
    ("What was the famous German fighter pilot",
     "The famous German fighter pilot known as the 'Red Baron' was named"),
    ("In what year did the Soviet Union officially dissolve",
     "The Soviet Union officially dissolved in the year"),
    ("Which ocean liner was sunk",
     "The ocean liner sunk by a German submarine in 1915 was the"),
    ("What city was divided by a famous wall",
     "The city divided by a famous wall from 1961 to 1989 was"),
    ("What war started when North Korea invaded",
     "The war that started when North Korea invaded South Korea in 1950 was"),
    ("What was the German plan to quickly defeat France",
     "The German plan to quickly defeat France was called"),
    ("What treaty officially ended World War I",
     "The treaty that officially ended World War I was the"),
    ("What event saw the US and UK airlift supplies",
     "The event in which the US and UK airlifted supplies into a blockaded "
     "German city was"),
    ("Who led the Russian Revolution",
     "The Russian Revolution that pulled Russia out of World War I was led by"),
    ("What fort was attacked",
     "The fort attacked to start the American Civil War was"),
    ("Who was the US President during the Cuban Missile Crisis",
     "The US President during the Cuban Missile Crisis was"),
    ("What empire was dissolved and replaced largely by Turkey",
     "The empire dissolved and replaced largely by Turkey after World War I was the"),
    ("Where did General Lee surrender",
     "General Lee surrendered to General Grant at"),
]

# --------------------------------------------------------------------------- #
# Similar-domain QA - test
# --------------------------------------------------------------------------- #
STEMS["SimdomQA_test"] = [
    ("What was the name of the first dog sent into space",
     "The first dog sent into space by the Soviets was named"),
    ("What new armored vehicle was introduced",
     "The new armored vehicle introduced by the British in World War I was"),
    ("What economic plan did the US use",
     "The economic plan the US used to help rebuild Western Europe was the"),
    ("What year did World War I start",
     "World War I started in the year"),
    ("What year did the American Civil War end",
     "The American Civil War ended in the year"),
    ("Which country launched Sputnik 1",
     "The country that launched Sputnik 1 was"),
    ("Where was Abraham Lincoln assassinated",
     "Abraham Lincoln was assassinated at"),
    ("What does the policy of",
     "In English, the policy of 'glasnost' means"),
    ("What agency was created in the US to lead the space race",
     "The agency created in the US to lead the space race was"),
    ("What communist leader took over Cuba",
     "The communist leader who took over Cuba in 1959 was"),
    ("Which Confederate general earned his famous nickname",
     "The Confederate general who earned his famous nickname at Bull Run was"),
    ("Who was the pilot of the downed U-2",
     "The pilot of the downed U-2 spy plane was"),
    ("What battle is considered the turning point",
     "The battle considered the turning point of the American Civil War is the"),
    ("What was the alliance of Britain, France, and Russia",
     "The alliance of Britain, France, and Russia was called"),
    ("What was the period of rebuilding the South",
     "The period of rebuilding the South after the Civil War was called"),
    ("What Constitutional amendment officially abolished slavery",
     "The Constitutional amendment that officially abolished slavery in the US "
     "was the"),
    ("What US policy aimed to prevent the spread of communism",
     "The US policy that aimed to prevent the spread of communism was"),
    ("What was the nickname for the Northern soldiers",
     "The nickname for the Northern soldiers was"),
    ("What was the first artificial satellite",
     "The first artificial satellite launched into space was"),
    ("What does 'perestroika' refer to",
     "The term 'perestroika' refers to"),
    ("What is the term for the area between two opposing trenches",
     "The term for the area between two opposing trenches is"),
    ("Which famous abolitionist led the raid",
     "The famous abolitionist who led the raid on Harper's Ferry before the war was"),
    ("Which global pandemic occurred",
     "The global pandemic that occurred toward the end of World War I was"),
    ("What was the first major land battle",
     "The first major land battle of the American Civil War was the"),
    ("What theory suggested that if one country fell",
     "The theory suggesting that if one country fell to communism others would "
     "follow was the"),
    ("Who was the commanding general of the Confederate Army",
     "The commanding general of the Confederate Army of Northern Virginia was"),
    ("Who was the leader of the Soviet Union during the Cuban Missile Crisis",
     "The leader of the Soviet Union during the Cuban Missile Crisis was"),
    ("The Zimmermann Telegram proposed an alliance",
     "The Zimmermann Telegram proposed an alliance between Germany and"),
    ("What passenger ship",
     "The passenger ship whose sinking helped turn American public opinion "
     "against Germany was the"),
    ("What deadly gas was notably used",
     "The deadly gas notably used as a weapon in World War I was"),
    ("What was the primary cause of the American Civil War",
     "The primary cause of the American Civil War was"),
    ("What international organization was created after World War I",
     "The international organization created after World War I to maintain "
     "peace was the"),
    ("What famous speech did Lincoln give",
     "The famous speech Lincoln gave to dedicate a military cemetery was"),
    ("What secret network helped enslaved people",
     "The secret network that helped enslaved people escape to the North was"),
    ("What ironclad ship fought for the Confederacy",
     "The ironclad ship that fought for the Confederacy was the"),
    ("What year did World War I end",
     "World War I ended in the year"),
    ("What color uniforms did the Union army",
     "The Union army primarily wore uniforms of the color"),
    ("What 1962 crisis brought the US and USSR",
     "The 1962 crisis that brought the US and USSR to the brink of nuclear war "
     "was the"),
    ("What was the British plan to attack the Ottoman Empire",
     "The British plan to attack the Ottoman Empire at the Dardanelles was called"),
    ("Which country switched from the Central Powers",
     "The country that switched from the Central Powers to the Allies in 1915 was"),
    ("What military alliance was formed by Western nations",
     "The military alliance formed by Western nations in 1949 was"),
    ("What major conflict involved US troops",
     "The major conflict that involved US troops fighting communist forces in "
     "Southeast Asia from the 1950s to 1970s was the"),
    ("What US Senator led controversial investigations",
     "The US Senator who led controversial investigations into alleged "
     "communists in the 1950s was"),
    ("Who assassinated Abraham Lincoln",
     "Abraham Lincoln was assassinated by"),
    ("Who was the leader of Germany during World War I",
     "The leader of Germany during World War I was"),
    ("Who was President of the United States during the American Civil War",
     "During the American Civil War, the President of the United States was"),
    ("Which side had more factories and railroads",
     "During the American Civil War, the side with more factories and railroads was"),
    ("What color uniforms did the Confederate army",
     "The Confederate army primarily wore uniforms of the color"),
    ("Which US President famously said",
     "The US President who famously said 'Mr. Gorbachev, tear down this wall' was"),
    ("Which two superpowers were the main rivals",
     "The two superpowers that were the main rivals in the Cold War were"),
]
