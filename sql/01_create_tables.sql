CREATE TABLE logements (
    id SERIAL PRIMARY KEY,
    surface FLOAT,
    nb_chambres INT,
    nb_salles_bain FLOAT,
    localisation VARCHAR(100),
    quartier VARCHAR(100),
    qualite VARCHAR(50),
    annee_construction INT,
    etat VARCHAR(50),
    prix_vente FLOAT
);

CREATE TABLE proprietes (
    id SERIAL PRIMARY KEY,
    logement_id INT NOT NULL UNIQUE,

    surface_totale FLOAT,
    nb_salles_bain_totales FLOAT,
    age_logement INT,
    anciennete_renovation INT,
    autres_indicateurs VARCHAR(255),

    CONSTRAINT fk_propriete_logement
        FOREIGN KEY (logement_id)
        REFERENCES logements(id)
        ON DELETE CASCADE
);

CREATE TABLE modeles (
    id SERIAL PRIMARY KEY,

    propriete_id INT NOT NULL,

    nom VARCHAR(50) NOT NULL,
    type_modele VARCHAR(50) NOT NULL,
    hyperparametres JSONB,
    performances JSONB,
    date_entrainement TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT fk_modele_propriete
        FOREIGN KEY (propriete_id)
        REFERENCES proprietes(id)
        ON DELETE CASCADE
);

CREATE TABLE evaluations (
    id SERIAL PRIMARY KEY,

    modele_id INT NOT NULL,

    mae FLOAT,
    rmse FLOAT,
    r2 FLOAT,
    date_evaluation TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT fk_evaluation_modele
        FOREIGN KEY (modele_id)
        REFERENCES modeles(id)
        ON DELETE CASCADE
);

CREATE TABLE features (
    id SERIAL PRIMARY KEY,

    nom VARCHAR(50) NOT NULL,
    type VARCHAR(50) NOT NULL,
    importance FLOAT
);

CREATE TABLE modele_features (
    modele_id INT NOT NULL,
    feature_id INT NOT NULL,

    PRIMARY KEY (modele_id, feature_id),

    CONSTRAINT fk_modele_features_modele
        FOREIGN KEY (modele_id)
        REFERENCES modeles(id)
        ON DELETE CASCADE,

    CONSTRAINT fk_modele_features_feature
        FOREIGN KEY (feature_id)
        REFERENCES features(id)
        ON DELETE CASCADE
);

CREATE TABLE predictions (
    id SERIAL PRIMARY KEY,

    logement_id INT NOT NULL,
    modele_id INT NOT NULL,

    prix_predit FLOAT,
    date_prediction TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT fk_prediction_logement
        FOREIGN KEY (logement_id)
        REFERENCES logements(id)
        ON DELETE CASCADE,

    CONSTRAINT fk_prediction_modele
        FOREIGN KEY (modele_id)
        REFERENCES modeles(id)
        ON DELETE CASCADE
);