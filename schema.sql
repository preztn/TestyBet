CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE,
    password TEXT NOT NULL,
    default_weights TEXT  -- Optionally store user-specific defaults as JSON
);

-- New predictions table:
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    team_defensive_rank INTEGER,
    player_position TEXT,
    player_stats TEXT,
    streaks TEXT,
    weights TEXT,
    probability_exceed_line REAL,
    will_exceed_line TEXT,
    projected_stat REAL,
    likelihood_score REAL,
    actual_outcome REAL,
    user_line REAL,       -- New column for storing the custom betting line
    win_loss TEXT,        -- "Win" or "Loss"
    FOREIGN KEY (user_id) REFERENCES users(id)
);

-- New table for storing external player stats
CREATE TABLE IF NOT EXISTS player_stats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id INTEGER,
    player_name TEXT,
    points REAL,
    rebounds REAL,
    assists REAL,
    team TEXT,
    date TEXT
    -- Add whatever columns you need, e.g., steals, blocks, etc.
);