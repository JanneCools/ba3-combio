# FlowSOM API
Deze repository biedt een FLowSOM API aan. 
Binnen de *src*-map bevinden zich verschillende packages en flowsom.py waarmee
een nieuw FlowSom object aangemaakt kan worden.

De gebruiker kan dan de fit-, predict- en fit_predict-methodes oproepen op dit 
object, waarbij data kan meegegeven worden om deze functies op uit te voeren.
Het FlowSom object zal een self-organising map maken van de data, 
deze visualiseren met behulp van een minimaal opspannende boom en er
metaclustering op toepassen. Dit gebeurt allemaal in de fit-methode.
De predict-methode zal de metaclusters voorspellen voor bepaalde data. 
Dit kan dezelfde data zijn als bij de fit-methode, maar mag ook andere data zijn.

Binnen het package *flowsom* zijn er nog enkele andere packages aanwezig, 
waarvan de werking hier ook wordt uitgelegd.

## Plotting
De package *plotting* bevat functies om plots te maken van de self-organising maps
en de minimal spanning trees. Deze functies moet de gebruiker zelf niet oproepen.
Het FlowSom object zal bij het uitvoeren van het algoritme automatisch plots 
laten genereren via deze package.

## Report
De package *report* zal gaandeweg een pdf aanvullen met verkregen plots wanneer
de SOM, MST en metaclusters gemaakt worden. Na het uitvoeren van fit 
(en eventueel ook de predict) kan de gebruiker de report-functie oproepen op het
FlowSom object om deze informatie te laten uitschrijven naar een meegegeven bestand.

## Util
De package *util* bevat enkele bestanden die gebruikt worden door het FlowSom object.
Zo zal reader.py de meegegeven input behandelen en omzetten naar een AnnData object.
Deze input kan een numpy array zijn, een pandas DataFrame, een bestandsnaam of een 
directory-naam.
Som.py zal de self-organising map creëren en trainen op de gekregen data.

