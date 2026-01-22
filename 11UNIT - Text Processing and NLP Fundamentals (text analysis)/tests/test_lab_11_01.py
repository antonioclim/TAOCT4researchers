from lab.lab_11_01_regex_string_ops import find_emails, parse_log_lines

def test_find_emails():
    s = "Write to a@b.com and cc to jane.doe@example.org."
    emails = find_emails(s)
    assert "a@b.com" in emails
    assert "jane.doe@example.org" in emails

def test_parse_log_lines_counts():
    lines = [
        "2025-01-10T09:15:00Z INFO user=alice msg=ok",
        "bad line",
        "2025-01-10T09:15:01Z WARN user=bob msg=warn",
    ]
    records = parse_log_lines(lines)
    assert len(records) == 2
