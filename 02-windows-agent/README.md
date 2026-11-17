# 2. Windows Agent + Workloads

**Zadužen:** Mihael Jakšić

Pozadinski program na Windows računalu koji:

- **mjeri sustav**: CPU (ukupno i po jezgri), RAM i disk I/O, svake sekunde
- **pokreće kontrolirane workloade**: testna opterećenja s poznatim tipom, jačinom i trajanjem, iz kojih nastaju podaci za učenje modela
- **upravlja procesima**: prioritet, CPU affinity, suspend/resume, gašenje
- **postavlja ograničenja**: Job Objects (CPU, memorija, broj procesa) i power plan
- **javlja događaje**: početak i kraj workloada, pokretanje procesa, promjene postavki

Podatke šalje platformi (dio 3), a naredbe prima od Prediction & Control dijela (dio 6).

Koristi `psutil` za mjerenje i upravljanje procesima, a `pywin32` za Job Objecte.
