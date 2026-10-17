"""Carrier text for the synthetic-speech generator (ml_tts.py). Data only; no logic.

A sentence is LEAD + CLAUSE + TAIL. LEADS/TAILS give the register (everyday / business / news); CLAUSES[lang][group] are
short natural clauses with one `{n}` slot where the WRITTEN number goes (the slot generators in ml_tts.py decide what
is written there, including nouns, units and context words; the clauses stay neutral). All text is original.

WANT[(lang, group)] = {span kind: regex}: a reading must match for that kind in that group (gender, case, joiner words),
because the candidate lists of ml_cands are written for any context while a clause fixes one.
"""

# --------------------------------------------------------------------------- leads / tails: (register, text)

LEADS = {
    "es": [("day", "Ayer por la tarde,"), ("day", "Oye, te cuento una cosa:"), ("day", "Pues mira, en casa nos dijeron que"),
           ("biz", "Según el informe trimestral,"), ("biz", "En la reunión de dirección se explicó que"),
           ("biz", "Para el próximo ejercicio,"), ("news", "Según informó la agencia esta mañana,"),
           ("news", "Fuentes oficiales confirmaron que"), ("day", ""), ("biz", ""), ("news", "")],
    "fr": [("day", "Hier après-midi,"), ("day", "Écoute, je te raconte :"), ("day", "Chez nous, on nous a dit que"),
           ("biz", "Selon le rapport trimestriel,"), ("biz", "Lors de la réunion de direction, on a précisé que"),
           ("biz", "Pour le prochain exercice,"), ("news", "Selon l'agence de presse ce matin,"),
           ("news", "Des sources officielles ont confirmé que"), ("day", ""), ("biz", ""), ("news", "")],
    "de": [("day", "Das ist gestern Nachmittag passiert:"), ("day", "Hör mal, ich erzähl dir was:"),
           ("day", "Bei uns zu Hause hieß es,"), ("biz", "Das steht im Quartalsbericht:"),
           ("biz", "In der Vorstandssitzung wurde erklärt,"), ("biz", "Wichtig für das kommende Geschäftsjahr:"),
           ("news", "Die Agentur meldete heute Morgen:"), ("news", "Offizielle Stellen bestätigten,"),
           ("day", ""), ("biz", ""), ("news", "")],
    "it": [("day", "Ieri pomeriggio,"), ("day", "Senti, ti racconto una cosa:"), ("day", "A casa nostra ci hanno detto che"),
           ("biz", "Secondo il rapporto trimestrale,"), ("biz", "Nella riunione di direzione si è spiegato che"),
           ("biz", "Per il prossimo esercizio,"), ("news", "Secondo quanto riferito dall'agenzia stamattina,"),
           ("news", "Fonti ufficiali hanno confermato che"), ("day", ""), ("biz", ""), ("news", "")],
    "pt": [("day", "Ontem à tarde,"), ("day", "Olha, vou te contar uma coisa:"), ("day", "Lá em casa disseram que"),
           ("biz", "Segundo o relatório trimestral,"), ("biz", "Na reunião da diretoria foi explicado que"),
           ("biz", "Para o próximo exercício,"), ("news", "Segundo a agência de notícias nesta manhã,"),
           ("news", "Fontes oficiais confirmaram que"), ("day", ""), ("biz", ""), ("news", "")],
    "nl": [("day", "Dit gebeurde gisteren aan het eind van de middag:"), ("day", "Luister, ik vertel je iets:"),
           ("day", "Bij ons thuis zeiden ze:"), ("biz", "Dit staat in het kwartaalverslag:"),
           ("biz", "In de directievergadering werd gezegd:"), ("biz", "Belangrijk voor het komende boekjaar:"),
           ("news", "Het persbureau meldde vanochtend:"), ("news", "Officiële bronnen bevestigden:"),
           ("day", ""), ("biz", ""), ("news", "")],
}

TAILS = {
    "es": [", y nadie lo esperaba", " y todos quedaron bastante sorprendidos", ", como ya habíamos comentado la semana pasada",
           ", al menos eso nos dijeron", "", "", ""],
    "fr": [", et personne ne s'y attendait", " et tout le monde a été surpris", ", comme nous l'avions dit la semaine dernière",
           ", du moins c'est ce qu'on nous a dit", "", "", ""],
    "de": [", und niemand hatte damit gerechnet", " und alle waren ziemlich überrascht", ", wie wir letzte Woche schon besprochen haben",
           ", zumindest hieß es so", "", "", ""],
    "it": [", e nessuno se lo aspettava", " e tutti sono rimasti molto sorpresi", ", come avevamo già detto la settimana scorsa",
           ", almeno così ci hanno detto", "", "", ""],
    "pt": [", e ninguém esperava isso", " e todos ficaram bastante surpresos", ", como já tínhamos comentado na semana passada",
           ", pelo menos foi o que nos disseram", "", "", ""],
    "nl": [", en niemand had dat verwacht", " en iedereen was behoorlijk verrast", ", zoals we vorige week al bespraken",
           ", dat zeiden ze tenminste", "", "", ""],
}

# --------------------------------------------------------------------------- clauses

CLAUSES = {
    "es": {
        "QTY": ["la cifra final fue de {n} según los datos publicados", "en total se contabilizaron {n} durante el año",
                "el último balance de la oficina de estadística indica {n}", "los responsables del proyecto mencionaron {n} en su informe",
                "el dato que más llamó la atención fue {n}"],
        "MONEY": ["el precio final del producto es de {n} con los impuestos incluidos", "el cliente pagó {n} por el servicio completo",
                  "la factura que recibimos ayer era de {n}", "la empresa destinó {n} a la reforma del edificio"],
        "TIME": ["la reunión empieza a las {n} en la sala principal", "el tren sale de la estación a las {n} sin falta",
                 "quedamos en la puerta del cine a las {n} para llegar a tiempo", "el director llegó a la oficina a las {n}"],
        "DATE_NUM": ["el contrato se firmó el {n} ante notario", "la entrega está prevista para el {n} en el almacén",
                     "el documento lleva fecha del {n} y el sello oficial"],
        "DATE_TXT": ["el contrato se firmó el {n} ante notario", "la entrega está prevista para el {n} en el almacén",
                     "nos vemos el {n} para cerrar el acuerdo"],
        "PHONE": ["para pedir cita puede llamar al {n} en horario de oficina", "apunta mi número de teléfono, es el {n}",
                  "el servicio de atención al cliente responde en el {n} todos los días"],
        "CODE": ["el pedido llegó con la referencia {n} en la etiqueta", "el cartel mencionaba {n} varias veces durante la charla",
                 "en la lista de productos aparece {n} como novedad", "el comentarista habló de {n} durante toda la tarde"],
        "YEAR": ["la empresa fue fundada en {n} por dos socios", "todo cambió en {n}, cuando llegó la nueva directiva",
                 "el edificio se terminó de construir en {n} después de muchos retrasos", "la ley entró en vigor en {n}"],
        "DECADE": ["esa música sonaba por todas partes durante {n}", "mis padres se conocieron en {n} en una fiesta del barrio",
                   "la economía creció mucho en {n} según los expertos"],
        "ORD_M": ["quedó en el {n} lugar de la clasificación general", "es el {n} piso del edificio, según el plano",
                  "fue el {n} intento del equipo en toda la temporada"],
        "ORD_F": ["es la {n} vez que visitamos esta ciudad", "se celebra la {n} edición del festival este fin de semana",
                  "ganó la {n} etapa de la carrera en solitario"],
        "ROMAN": ["el profesor explicó {n} con mucho detalle en clase", "los alumnos estudiaron {n} durante todo el trimestre"],
        "RANGE_FROM": ["la visita dura de {n} según el programa", "el equipo trabajó de {n} cada día"],
        "RANGE_BETWEEN": ["asistieron entre {n} a la conferencia", "el coste estimado está entre {n} por unidad"],
        "TEMP": ["por la noche la temperatura bajará hasta {n} en la sierra", "el termómetro marcaba {n} a primera hora"],
        "FRAC": ["la receta lleva {n} para que quede bien jugosa", "el depósito está lleno hasta {n} según el indicador"],
        "AMP": ["el equipo comercial presentó {n} en la feria del sector", "esa marca, {n}, anunció sus resultados hoy"],
        "VERSION": ["actualice el programa a la versión {n} antes de continuar", "el servidor responde en la dirección {n} desde esta mañana"],
    },
    "fr": {
        "QTY": ["le chiffre final était de {n} selon les données publiées", "au total, on a compté {n} pendant l'année",
                "le dernier bilan de l'institut de statistique indique {n}", "les responsables du projet ont cité {n} dans leur rapport",
                "le chiffre qui a le plus frappé était {n}"],
        "MONEY": ["le prix final du produit est de {n} toutes taxes comprises", "le client a payé {n} pour le service complet",
                  "la facture que nous avons reçue hier était de {n}", "l'entreprise a consacré {n} à la rénovation du bâtiment"],
        "TIME": ["la réunion commence à {n} dans la grande salle", "le train part de la gare à {n} précises",
                 "on se retrouve devant le cinéma à {n} pour ne pas être en retard", "le directeur est arrivé au bureau à {n}"],
        "DATE_NUM": ["le contrat a été signé le {n} chez le notaire", "la livraison est prévue pour le {n} à l'entrepôt",
                     "le document porte la date du {n} et le cachet officiel"],
        "DATE_TXT": ["le contrat a été signé le {n} chez le notaire", "la livraison est prévue pour le {n} à l'entrepôt",
                     "on se voit le {n} pour conclure l'accord"],
        "PHONE": ["pour prendre rendez-vous, vous pouvez appeler le {n} aux heures de bureau",
                  "note mon numéro de téléphone, c'est le {n}", "le service client répond au {n} tous les jours"],
        "CODE": ["la commande est arrivée avec la référence {n} sur l'étiquette", "l'affiche mentionnait {n} plusieurs fois pendant l'exposé",
                 "dans la liste des produits, {n} apparaît comme nouveauté", "le commentateur a parlé de {n} toute la soirée"],
        "YEAR": ["l'entreprise a été fondée en {n} par deux associés", "tout a changé en {n}, quand la nouvelle direction est arrivée",
                 "le bâtiment a été achevé en {n} après beaucoup de retards", "la loi est entrée en vigueur en {n}"],
        "DECADE": ["cette musique passait partout pendant {n}", "mes parents se sont rencontrés pendant {n} lors d'une fête de quartier",
                   "l'économie a beaucoup progressé pendant {n} selon les experts"],
        "ORD_M": ["il a terminé au {n} rang du classement général", "c'est le {n} étage de l'immeuble, d'après le plan",
                  "c'était le {n} essai de l'équipe de toute la saison"],
        "ORD_F": ["c'est la {n} fois que nous visitons cette ville", "on célèbre la {n} édition du festival ce week-end",
                  "elle a remporté la {n} étape de la course en solitaire"],
        "ROMAN": ["le professeur a expliqué {n} très en détail en classe", "les élèves ont étudié {n} pendant tout le trimestre"],
        "RANGE_FROM": ["la visite dure de {n} selon le programme", "l'équipe a travaillé de {n} chaque jour"],
        "RANGE_BETWEEN": ["il y avait entre {n} à la conférence", "le coût estimé se situe entre {n} par unité"],
        "TEMP": ["la nuit, la température descendra jusqu'à {n} dans les montagnes", "le thermomètre indiquait {n} tôt le matin"],
        "FRAC": ["la recette demande {n} pour que ce soit bien moelleux", "le réservoir est rempli jusqu'à {n} d'après l'indicateur"],
        "AMP": ["l'équipe commerciale a présenté {n} au salon professionnel", "cette marque, {n}, a annoncé ses résultats aujourd'hui"],
        "VERSION": ["mettez le logiciel à jour vers la version {n} avant de continuer", "le serveur répond à l'adresse {n} depuis ce matin"],
    },
    "de": {
        "QTY": ["die Endzahl lag laut den veröffentlichten Daten bei {n}", "insgesamt wurden im Laufe des Jahres {n} gezählt",
                "die letzte Bilanz des Statistikamts nennt {n}", "die Projektverantwortlichen nannten {n} in ihrem Bericht",
                "die Zahl, die am meisten auffiel, war {n}"],
        "MONEY": ["der Endpreis des Produkts beträgt {n} inklusive aller Steuern", "der Kunde zahlte {n} für die gesamte Leistung",
                  "die Rechnung, die wir gestern bekamen, lautete auf {n}", "die Firma steckte {n} in die Sanierung des Gebäudes"],
        "TIME": ["die Besprechung beginnt um {n} im großen Saal", "der Zug fährt um {n} pünktlich vom Bahnhof ab",
                 "wir treffen uns um {n} vor dem Kino, um rechtzeitig da zu sein", "der Direktor kam um {n} ins Büro"],
        "DATE_NUM": ["der Vertrag wurde am {n} beim Notar unterschrieben", "die Lieferung ist am {n} im Lager vorgesehen",
                     "das Dokument trägt das Datum vom {n} und den amtlichen Stempel"],
        "DATE_TXT": ["der Vertrag wurde am {n} beim Notar unterschrieben", "die Lieferung ist am {n} im Lager vorgesehen",
                     "wir sehen uns am {n}, um die Vereinbarung abzuschließen"],
        "PHONE": ["für einen Termin können Sie die {n} während der Bürozeiten anrufen", "schreib dir meine Telefonnummer auf, sie lautet {n}",
                  "der Kundendienst ist täglich unter {n} erreichbar"],
        "CODE": ["die Bestellung kam mit der Referenz {n} auf dem Etikett an", "auf dem Plakat stand {n} während des Vortrags mehrmals",
                 "in der Produktliste taucht {n} als Neuheit auf", "der Kommentator sprach den ganzen Abend über {n}"],
        "YEAR": ["die Firma wurde {n} von zwei Teilhabern gegründet", "alles änderte sich {n}, als die neue Leitung kam",
                 "das Gebäude wurde {n} nach vielen Verzögerungen fertig", "das Gesetz trat {n} in Kraft"],
        "DECADE": ["diese Musik lief überall in {n}", "meine Eltern lernten sich in {n} auf einem Nachbarschaftsfest kennen",
                   "die Wirtschaft wuchs in {n} laut den Fachleuten stark"],
        "ORD_M": ["er landete auf dem {n} Platz der Gesamtwertung", "wir wohnen im {n} Stock des Gebäudes, laut Plan", "es war beim {n} Versuch der Mannschaft in der ganzen Saison"],
        "ORD_F": ["das ist die {n} Runde, die wir in dieser Stadt drehen", "an diesem Wochenende findet die {n} Auflage des Festivals statt", "sie gewann die {n} Etappe des Rennens im Alleingang"],
        "ROMAN": ["der Lehrer erklärte {n} im Unterricht sehr ausführlich", "die Schüler beschäftigten sich das ganze Quartal mit {n}"],
        "RANGE_FROM": ["der Besuch dauert laut Programm von {n}", "das Team arbeitete jeden Tag von {n}"],
        "RANGE_BETWEEN": ["zur Konferenz kamen zwischen {n}", "die geschätzten Kosten liegen zwischen {n} pro Stück"],
        "TEMP": ["nachts sinkt die Temperatur im Gebirge bis auf {n}", "das Thermometer zeigte am frühen Morgen {n}"],
        "FRAC": ["das Rezept braucht {n}, damit es schön saftig wird", "der Tank ist laut Anzeige bis {n} gefüllt"],
        "AMP": ["das Vertriebsteam stellte {n} auf der Fachmesse vor", "diese Marke, {n}, gab heute ihre Ergebnisse bekannt"],
        "VERSION": ["aktualisieren Sie das Programm auf Version {n}, bevor Sie weitermachen", "der Server antwortet seit heute Morgen unter der Adresse {n}"],
    },
    "it": {
        "QTY": ["la cifra finale è stata di {n} secondo i dati pubblicati", "in totale ne sono stati contati {n} nel corso dell'anno",
                "l'ultimo bilancio dell'istituto di statistica indica {n}", "i responsabili del progetto hanno citato {n} nel loro rapporto",
                "il dato che ha colpito di più era {n}"],
        "MONEY": ["il prezzo finale del prodotto è di {n} con le tasse incluse", "il cliente ha pagato {n} per il servizio completo",
                  "la fattura che abbiamo ricevuto ieri era di {n}", "l'azienda ha destinato {n} alla ristrutturazione dell'edificio"],
        "TIME": ["la riunione comincia alle {n} nella sala grande", "il treno parte dalla stazione alle {n} in punto",
                 "ci vediamo davanti al cinema alle {n} per arrivare in tempo", "il direttore è arrivato in ufficio alle {n}"],
        "DATE_NUM": ["il contratto è stato firmato il {n} davanti al notaio", "la consegna è prevista per il {n} in magazzino",
                     "il documento porta la data del {n} e il timbro ufficiale"],
        "DATE_TXT": ["il contratto è stato firmato il {n} davanti al notaio", "la consegna è prevista per il {n} in magazzino",
                     "ci vediamo il {n} per chiudere l'accordo"],
        "PHONE": ["per prendere appuntamento può chiamare il {n} in orario d'ufficio", "segnati il mio numero di telefono, è il {n}",
                  "il servizio clienti risponde al {n} tutti i giorni"],
        "CODE": ["l'ordine è arrivato con il riferimento {n} sull'etichetta", "il cartellone citava {n} più volte durante l'intervento",
                 "nell'elenco dei prodotti compare {n} come novità", "il commentatore ha parlato di {n} per tutta la serata"],
        "YEAR": ["l'azienda è stata fondata nel {n} da due soci", "tutto è cambiato nel {n}, quando è arrivata la nuova dirigenza",
                 "l'edificio è stato completato nel {n} dopo molti ritardi", "la legge è entrata in vigore nel {n}"],
        "DECADE": ["quella musica suonava dappertutto durante {n}", "i miei genitori si sono conosciuti durante {n} a una festa di quartiere",
                   "l'economia è cresciuta molto durante {n} secondo gli esperti"],
        "ORD_M": ["è arrivato al {n} posto della classifica generale", "è il {n} piano dell'edificio, secondo la pianta",
                  "è stato il {n} tentativo della squadra in tutta la stagione"],
        "ORD_F": ["è la {n} volta che visitiamo questa città", "si svolge la {n} edizione del festival questo fine settimana",
                  "ha vinto la {n} tappa della corsa in solitaria"],
        "ROMAN": ["il professore ha spiegato {n} con molti dettagli in classe", "gli studenti hanno studiato {n} per tutto il trimestre"],
        "RANGE_FROM": ["la visita dura da {n} secondo il programma", "la squadra ha lavorato da {n} ogni giorno"],
        "RANGE_BETWEEN": ["alla conferenza c'erano tra {n} secondo gli organizzatori", "il costo stimato è tra {n} per unità"],
        "TEMP": ["di notte la temperatura scenderà fino a {n} in montagna", "il termometro segnava {n} di prima mattina"],
        "FRAC": ["la ricetta richiede {n} perché venga bella morbida", "il serbatoio è pieno fino a {n} secondo l'indicatore"],
        "AMP": ["la squadra commerciale ha presentato {n} alla fiera di settore", "quel marchio, {n}, ha annunciato oggi i risultati"],
        "VERSION": ["aggiorni il programma alla versione {n} prima di continuare", "il server risponde all'indirizzo {n} da questa mattina"],
    },
    "pt": {
        "QTY": ["o número final foi de {n} segundo os dados publicados", "ao todo foram contados {n} durante o ano",
                "o último balanço do instituto de estatística indica {n}", "os responsáveis pelo projeto citaram {n} no relatório",
                "o dado que mais chamou a atenção foi {n}"],
        "MONEY": ["o preço final do produto é de {n} com os impostos incluídos", "o cliente pagou {n} pelo serviço completo",
                  "a fatura que recebemos ontem era de {n}", "a empresa destinou {n} à reforma do prédio"],
        "TIME": ["a reunião começa às {n} na sala principal", "o trem sai da estação às {n} sem falta",
                 "vamos nos encontrar na porta do cinema às {n} para chegar a tempo", "o diretor chegou ao escritório às {n}"],
        "DATE_NUM": ["o contrato foi assinado no dia {n} no cartório", "a entrega está prevista para o dia {n} no armazém",
                     "o documento tem a data de {n} e o carimbo oficial"],
        "DATE_TXT": ["o contrato foi assinado no dia {n} no cartório", "a entrega está prevista para {n} no armazém",
                     "nos vemos em {n} para fechar o acordo"],
        "PHONE": ["para marcar uma consulta, ligue para o {n} em horário comercial", "anota meu número de telefone, é o {n}",
                  "o atendimento ao cliente responde no {n} todos os dias"],
        "CODE": ["o pedido chegou com a referência {n} na etiqueta", "o cartaz mencionava {n} várias vezes durante a palestra",
                 "na lista de produtos aparece {n} como novidade", "o comentarista falou de {n} a tarde inteira"],
        "YEAR": ["a empresa foi fundada em {n} por dois sócios", "tudo mudou em {n}, quando chegou a nova diretoria",
                 "o prédio ficou pronto em {n} depois de muitos atrasos", "a lei entrou em vigor em {n}"],
        "DECADE": ["essa música tocava em todo lugar durante {n}", "meus pais se conheceram durante {n} numa festa do bairro",
                   "a economia cresceu muito durante {n} segundo os especialistas"],
        "ORD_M": ["ele ficou em {n} lugar na classificação geral", "é o {n} andar do prédio, segundo a planta",
                  "foi o {n} tempo da equipe em toda a temporada"],
        "ORD_F": ["é a {n} vez que visitamos esta cidade", "acontece a {n} edição do festival neste fim de semana",
                  "ela venceu a {n} etapa da corrida sozinha"],
        "ROMAN": ["o professor explicou {n} com muitos detalhes na aula", "os alunos estudaram {n} durante todo o trimestre"],
        "RANGE_FROM": ["a visita dura de {n} segundo o programa", "a equipe trabalhou de {n} todos os dias"],
        "RANGE_BETWEEN": ["compareceram entre {n} à conferência", "o custo estimado fica entre {n} por unidade"],
        "TEMP": ["à noite a temperatura vai cair até {n} na serra", "o termômetro marcava {n} logo cedo"],
        "FRAC": ["a receita leva {n} para ficar bem macia", "o tanque está cheio até {n} segundo o indicador"],
        "AMP": ["a equipe comercial apresentou {n} na feira do setor", "essa marca, {n}, anunciou os resultados hoje"],
        "VERSION": ["atualize o programa para a versão {n} antes de continuar", "o servidor responde no endereço {n} desde esta manhã"],
    },
    "nl": {
        "QTY": ["het eindcijfer was volgens de gepubliceerde gegevens {n}", "in totaal zijn er in de loop van het jaar {n} geteld",
                "de laatste balans van het statistiekbureau noemt {n}", "de projectverantwoordelijken noemden {n} in hun verslag",
                "het getal dat het meest opviel was {n}"],
        "MONEY": ["de eindprijs van het product is {n} inclusief alle belastingen", "de klant betaalde {n} voor de volledige dienst",
                  "de rekening die we gisteren kregen was voor {n}", "het bedrijf stak {n} in de renovatie van het gebouw"],
        "TIME": ["de vergadering begint om {n} in de grote zaal", "de trein vertrekt om {n} stipt van het station",
                 "we spreken om {n} af voor de bioscoop om op tijd te zijn", "de directeur kwam om {n} op kantoor"],
        "DATE_NUM": ["het contract is op {n} bij de notaris getekend", "de levering staat gepland voor {n} in het magazijn",
                     "het document draagt de datum van {n} en het officiële stempel"],
        "DATE_TXT": ["het contract is op {n} bij de notaris getekend", "de levering staat gepland op {n} in het magazijn",
                     "we zien elkaar op {n} om de afspraak af te ronden"],
        "PHONE": ["voor een afspraak kunt u tijdens kantooruren bellen naar {n}", "schrijf mijn telefoonnummer maar op, het is {n}",
                  "de klantenservice is elke dag bereikbaar op {n}"],
        "CODE": ["de bestelling kwam aan met de referentie {n} op het etiket", "op de poster stond tijdens de lezing meerdere keren {n}",
                 "in de productlijst staat {n} als nieuwtje", "de commentator had de hele avond over {n}"],
        "YEAR": ["het bedrijf werd in {n} opgericht door twee vennoten", "alles veranderde in {n}, toen de nieuwe directie kwam",
                 "het gebouw was in {n} klaar na veel vertraging", "de wet trad in {n} in werking"],
        "DECADE": ["die muziek klonk overal in {n}", "mijn ouders leerden elkaar in {n} kennen op een buurtfeest",
                   "de economie groeide in {n} volgens de experts sterk"],
        "ORD_M": ["hij eindigde op de {n} plaats in het algemeen klassement", "het is de {n} verdieping van het gebouw volgens de plattegrond",
                  "het was de {n} poging van het team in het hele seizoen"],
        "ORD_F": ["het is de {n} keer dat we deze stad bezoeken", "dit weekend vindt de {n} editie van het festival plaats",
                  "ze won de {n} etappe van de wedstrijd in haar eentje"],
        "ROMAN": ["de leraar legde {n} uitgebreid uit in de les", "de leerlingen bestudeerden {n} het hele kwartaal"],
        "RANGE_FROM": ["het bezoek duurt volgens het programma van {n}", "het team werkte elke dag van {n}"],
        "RANGE_BETWEEN": ["er kwamen tussen {n} naar de conferentie", "de geschatte kosten liggen tussen {n} per stuk"],
        "TEMP": ["'s nachts daalt de temperatuur in de bergen tot {n}", "de thermometer gaf vroeg in de ochtend {n} aan"],
        "FRAC": ["het recept vraagt om {n} zodat het lekker sappig wordt", "de tank is volgens de meter tot {n} gevuld"],
        "AMP": ["het verkoopteam presenteerde {n} op de vakbeurs", "dat merk, {n}, maakte vandaag zijn resultaten bekend"],
        "VERSION": ["werk het programma bij naar versie {n} voordat u verdergaat", "de server reageert sinds vanochtend op het adres {n}"],
    },
}

# --------------------------------------------------------------------------- joiner / case constraints per (lang, group)

_FROM = {"es": r"^ ?(a|hasta)\b", "pt": r"^ ?(a|até)\b", "it": r"^ ?(a|fino a)\b", "fr": r"^ ?à\b",
         "de": r"^ ?bis\b", "nl": r"^ ?(tot|tot en met)\b"}
_BETWEEN = {"es": r"^ ?y\b", "pt": r"^ ?e\b", "it": r"^ ?e\b", "fr": r"^ ?et\b", "de": r"^ ?und\b", "nl": r"^ ?en\b"}
WANT = {}
for _l in ("es", "pt", "it", "fr", "de", "nl"):
    WANT[(_l, "RANGE_FROM")] = {"range": _FROM[_l]}
    WANT[(_l, "RANGE_BETWEEN")] = {"range": _BETWEEN[_l]}
# German "am 12.05.2020" / "am 3. Mai": dative ordinal; the dotted ordinal before a noun ("der 3. Platz") is read per gender in ml_tts
WANT[("de", "DATE_NUM")] = {"date0": r"en$", "date1": r"(?:^[A-ZÄÖÜ]\w+$|en$)"}  # "am 12.05.2020": dative day ordinal
WANT[("de", "DATE_TXT")] = {"ord": r"en$"}  # "am 3. Mai"
WANT[("de", "ORD_M")] = {"ord": r"en$"}  # "auf dem dritten Platz"
WANT[("de", "ORD_F")] = {"ord": r"e$"}  # "die dritte Runde"

# nouns for counted quantities: masculine plurals (agreement-neutral for the digit readings the app builds)
NOUNS = {
    "es": ["empleados", "clientes", "pasajeros", "kilómetros de carretera", "vecinos", "ordenadores", "alumnos", "pedidos"],
    "fr": ["employés", "clients", "passagers", "kilomètres de route", "voisins", "ordinateurs", "élèves", "colis"],
    "de": ["Mitarbeiter", "Kunden", "Fahrgäste", "Kilometer Straße", "Nachbarn", "Computer", "Schüler", "Pakete"],
    "it": ["dipendenti", "clienti", "passeggeri", "chilometri di strada", "vicini", "computer", "studenti", "pacchi"],
    "pt": ["funcionários", "clientes", "passageiros", "quilômetros de estrada", "vizinhos", "computadores", "alunos", "pedidos"],
    "nl": ["medewerkers", "klanten", "passagiers", "kilometer weg", "buren", "computers", "leerlingen", "pakketten"],
}
# decimals read as a plain quantity
DEC_NOUNS = {"es": ["puntos", "veces", "litros"], "fr": ["points", "fois", "litres"], "de": ["Punkte", "Mal", "Liter"],
             "it": ["punti", "volte", "litri"], "pt": ["pontos", "vezes", "litros"], "nl": ["punten", "keer", "liter"]}
# "of money" nouns after a scale word: es "millones de euros"
SCALE = {  # (plural scale word, preposition, big scale word)
    "es": ("millones", "de", "mil millones"), "fr": ("millions", "d'", "milliards"), "de": ("Millionen", "", "Milliarden"),
    "it": ("milioni", "di", "miliardi"), "pt": ("milhões", "de", "bilhões"), "nl": ("miljoen", "", "miljard"),
}
SCALE_NOUN = {"es": ["euros", "dólares", "personas"], "fr": ["euros", "dollars", "personnes"], "de": ["Euro", "Dollar", "Menschen"],
              "it": ["euro", "dollari", "persone"], "pt": ["euros", "dólares", "pessoas"], "nl": ["euro", "dollar", "mensen"]}
FRAC_NOUN = {"es": ["litro", "hora", "kilo"], "fr": ["litre", "heure", "kilo"], "de": ["Liter", "Kilo", "Meter"],
             "it": ["litro", "ora", "chilo"], "pt": ["litro", "hora", "quilo"], "nl": ["liter", "kilo", "meter"]}
FRAC_NOUN_PL = {"es": ["litros", "horas", "kilos"], "fr": ["litres", "heures", "kilos"], "de": ["Liter", "Stunden", "Kilo"],
                "it": ["litri", "ore", "chili"], "pt": ["litros", "horas", "quilos"], "nl": ["liter", "uur", "kilo"]}
# names that take a roman numeral (NAME_ROMAN) and context words (ROMAN_BEFORE / ROMAN_AFTER of ml_cands)
ROMAN_NAMES = {"es": ["Luis", "Felipe", "Carlos", "Alfonso", "Juan Pablo"], "fr": ["Louis", "Charles", "Henri", "Napoléon", "Jean-Paul"],
               "de": ["Ludwig", "Friedrich", "Heinrich", "Karl", "Wilhelm"], "it": ["Luigi", "Pio", "Giovanni", "Enrico", "Carlo"],
               "pt": ["Pedro", "Luís", "Afonso", "Henrique", "João"], "nl": ["Willem", "Lodewijk", "Karel", "Hendrik", "Filips"]}
ROMAN_CTX = {"es": ["siglo", "capítulo", "tomo", "acto"], "fr": ["chapitre", "tome", "acte", "livre"],
             "de": ["Kapitel", "Band", "Teil", "Akt"], "it": ["secolo", "capitolo", "tomo", "atto"],
             "pt": ["século", "capítulo", "tomo", "ato"], "nl": ["hoofdstuk", "deel", "boek", "akte"]}
ROMAN_AFTER = {"es": "siglo", "fr": "siècle", "de": "Jahrhundert", "it": "secolo", "pt": "século", "nl": "eeuw"}
ROMAN_BEFORE_PL = {"es": "siglo", "pt": "século", "it": "secolo"}
# decade phrases: the context ml_cands requires, {d} = the two digits
DECADE_FORMS = {
    "es": ["los años {d}", "los años '{d}"], "pt": ["os anos {d}", "anos {d}"], "fr": ["les années {d}", "les années '{d}"],
    "it": ["gli anni {d}", "anni '{d}"], "nl": ["de jaren {d}", "de jaren '{d}"],
    "de": ["den {d}ern", "den {d}er Jahren"],
}
AMP_ACRONYMS = ["Q&A", "R&D", "M&A", "P&L", "B&B", "Q&C", "H&M", "AT&T", "P&G", "S&P", "I&D", "D&G"]
AMP_NAMES = ["Johnson & Johnson", "Marks & Spencer", "Procter & Gamble", "Ernst & Young", "Smith & Wesson", "Barnes & Noble"]
CODE_WORDS = ["COVID-19", "Boeing 737", "iPhone 15", "Windows 11", "Ubuntu 22", "Euro 2024", "Mercedes 300", "Citroën 2"]
CODE_LETTER_NUM = ["A4", "G7", "B52", "F1", "M5", "Q3", "H2", "C4", "A380", "E3", "G20", "U2", "R2", "X5", "T3"]
CODE_NUM_LETTER = ["5G", "4K", "3D", "2D", "4G", "1D", "8K", "6G"]
