# Dylan Kenneth Eliot

'''

This code was modeled for web apps like ipchat.in/kubernetes via katavm so the stack remained secure regardless of runtime deadlocking and hardware throttling.
 At no point does it make it easier to violate each tech stack but harder.

 If they don't have the room number or key strata, data lost. Same goes for the "username" assignment.

 If fleshed out further, this can be used to keep chat apps secure more than RSA/TSL/SSL could.
 '''

import os
import secrets
import sys
from pathlib import Path

BITS = 512
OUTPUT = Path("credentials.txt")


def generate_credentials():
    # Independent CSPRNG outputs: 512 bits each.
    seed = secrets.randbits(BITS)
    username = "u" + str(secrets.randbits(BITS))
    return seed, username


def main():
    seed, username = generate_credentials()
    content = f"seed={seed}\nusername={username}\n"

    # Create the file with owner-only permissions on POSIX systems.
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    fd = os.open(OUTPUT, flags, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"Created: {OUTPUT.resolve()}")
    print(f"Seed digits: {len(str(seed))} (512 random bits)")
    print(f"Username digits: {len(username) - 1} (512 random bits, prefixed with 'u')")
    print("Keep the file private; do not reuse these values as passwords or cryptographic keys.")


if __name__ == "__main__":
    try:
        main()
    except FileExistsError:
        print(f"Refusing to overwrite existing {OUTPUT}. Move it aside or choose another output filename.", file=sys.stderr)
        sys.exit(1)
