"""
Pure logic for parsing UniProt FASTA headers and records.

No network calls, no file I/O.
"""

from collections.abc import Iterable, Iterator


def parse_fasta_accession(header_line: str) -> str | None:
    """'>sp|Q6GZX4|001R_FRG3G Putative transcription factor 001R OS=...'
    -> 'Q6GZX4'. Returns None for malformed headers (missing '>' prefix or
    fewer than 3 '|'-delimited fields)."""
    if not header_line.startswith(">"):
        return None
    fields = header_line[1:].split("|")
    if len(fields) < 2:
        return None
    return fields[1]


def iter_fasta_records(lines: Iterable[str]) -> Iterator[tuple[str, str]]:
    """Streams (accession, sequence) pairs from FASTA-format lines, without
    holding the whole file in memory. Sequence lines are concatenated until
    the next '>' header or end of input."""
    accession: str | None = None
    seq_parts: list[str] = []
    for line in lines:
        line = line.rstrip("\n")
        if line.startswith(">"):
            if accession is not None:
                yield accession, "".join(seq_parts)
            accession = parse_fasta_accession(line)
            seq_parts = []
        elif accession is not None:
            seq_parts.append(line)
    if accession is not None:
        yield accession, "".join(seq_parts)


def extract_sequences(lines: Iterable[str], wanted_accessions: set[str]) -> dict[str, str]:
    """Streams through lines, keeping only sequences for wanted_accessions --
    the memory-bounded way to pull a handful of entries out of a
    multi-hundred-MB, all-organism FASTA file."""
    result: dict[str, str] = {}
    for accession, sequence in iter_fasta_records(lines):
        if accession in wanted_accessions:
            result[accession] = sequence
    return result
