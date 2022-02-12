# Projekt_WdSI_MP
 
Po uruchomieniu, program zapisuje do zmiennych globalnych informacje o ścieżkach dostępu do folderu nadrzędnego, a także poszczególnych potrzebnych później podfolderów. Te informacje są niezbędne do dalszych kroków.


Przy pomocy funkcji "import_data(path, set):" na podstawie ścieżek dostępu (argument 'path'), oraz informacji, że interesuje nas zbiór treningowy (argument 'set' = "train") program pobiera niezbędne dane o zdjęciach z pliku .xml i zapisuje dane do listy słowników. Są to m.in. ścieżka dostępu do odpowiadającego pliku .png (pole 'image'), liczba wszystkich znaków na zdjęciu (pole 'signs_number') oraz liczba znaków ograniczeń prędkości na zdjęciu (pole 'limits_number').
Dla każdego znaku wykrytego na zdjęciu, funkcja ta zapisuje również współrzędne xmin, xmax, ymin, ymax ramki, w której znak się znajduje, oraz informację, czy jest to znak speedlimit (pole 'limit_flag{j}' = 1, jeśli jest to ograniczenie prędkości, w przeciwnym wypadku 'limit_flag{j}' = 0). Funkcji można używać również dla zbioru testowego (argument 'set' = "test"), natomiast służyło to jedynie do sprawdzania poprawności działania algorytmu na końcowym etapie jego tworzenia, i nie jest wykorzystywane przy zadaniu klasyfikacji.

Następnie funkcja "crop_images(database)" przyjmuje jako argument stworzoną przez funkcję "import_data(path, set):" listę słowników. Mając wszystkie wymienione w poprzednim akapicie dane, dla każdego elementu listy (czyli de facto dla każdego pliku .png) funkcja 'wycina' fragmenty zdjęć z interesującymi nas znakami speedlimit i nadaje im 'label' = 1 oraz fragmenty bez interesujących nas znaków i nadaje im 'label' = 0. Wynikiem działania funkcji jest lista słowników o polach 'image' oraz 'label' zawierających kolejno wycinek zdjęcia (użyta została funkcja cv2.imread(path)) oraz informację o tym, czy w danym wycinku zdjęcia znajduje się interesujący nas znak.

Zdaję sobię sprawę, że można było obie powyższe funkcjonalności zawrzeć w jednej funkcji, jednak dla przejrzystości kodu, postanowiłem zrealizować je w dwóch osobnych funkcjach.

Zwrócona przez funkcję "crop_images(database)" lista słowników trafia następnie na używane wcześniej w zadaniu laboratoryjnym funkcje "learn_bovw(data)", "extract_features(data)". W wyniku działania tych dwóch funkcji otrzymujemy naszą listę wzbogaconą o pole ['desc'] zawierające deskryptory dla każdego z wycinków. Taką listę przyjmuje funkcja (również używana wcześniej do zadania laboratoryjnego) "train(data)". 
W efekcie otrzymujemy model ('rf') nauczony, wytrenowany przy pomocy RandomForestClassifier na podstawie deskryptorów wycinków zdjęć (NIE samych w sobie wycinków zdjęć!) oraz informacji, czy na danym wycinku znajduje się interesujący nas znak, czy też nie.

Możemy przejść do fazy pobierania danych testowych i predykcji dla zadanych fragmentów obrazów.

Po zakończeniu uczenia modelu, wyświetli się komunikat "model_trained". Wówczas rozpoczyna działanie funkcja "input_2()" i w przypadku obsługi z klawiatury należy wpisać (zgodnie z instrukcją do zadania projektowego):
- "classify" by wejść do realizacji zadania klasyfikacji
- liczbę plików (n), z których chcemy 'wyciąć' fragmenty obrazu
a następnie kolejno:
- dla n plików:
	- nazwę pliku .png
	- liczbę (m) wycinków w obrębie danego pliku .png
	- dla m wycinków:
		- współrzędne wycinka w kolejności xmin, xmax, ymin, ymax (oddzielając je spacjami, zatwierdzajac enterem 		po wpisaniu wszystkich czterech)

Funkcja "input_2()" na podstawie wpisanych danych 'wycina' określone przez użytkownika ramki, z określonych obrazków i tworzy z nich listę jednoelementowych słowników o polu 'image' zawierającym wycięty fragment obrazu. Lista ta jest nastepnie zwracana przez funkcję. Warto zauważyć, że w przeciwieństwie do etapu trenowania, nasze dane w tym przypadku nie mają 'labela' określającego, czy na danym wycinku znajduje się znak, czy nie.

Zwrócona lista trafia na funkcję "extract features(data)", gdzie wycinki otrzymują deskryptory, a nastepnie na funkcję "predict(rf, data)", która przyjmuje również wytrenowany wcześniej model "rf".

W wyniku predykcji, jeśli wykryty został znak ograniczenia prędkości (rf.predict(sample['desc']) = 1), wyświetlony zostaje komunikat "speedlimit". 
W przeciwnym wypadku wyświetlony zostaje komunikat "other".

Na tym działanie progrmau kończy się.


