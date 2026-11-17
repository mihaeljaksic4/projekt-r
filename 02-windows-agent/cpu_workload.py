#stvaranje opterećenja


import argparse
import multiprocessing as mp
import time


def radnik(kraj):
    while time.time() < kraj:
        pass

    
def main():
    parser = argparse.ArgumentParser(description="CPU workload")
    parser.add_argument("--brJezgri", type=int, default=1, help="Broj jezgri koje opterećujemo")
    parser.add_argument("--trajanje", type=float, default=10, help="trajanje u sekundama")
    args = parser.parse_args()
    
    kraj = time.time() + args.trajanje
    radnici = [mp.Process(target=radnik, args=(kraj,)) for _ in range (args.brJezgri)]
    
    print(f"Start: {args.brJezgri} radnika, {args.trajanje} s")
    for r in radnici:
        r.start()
    for r in radnici:
        r.join()
    print("Kraj.")

    
if __name__ == "__main__":
    main()