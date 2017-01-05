--
-- PostgreSQL database dump
--

-- Dumped from database version 9.5.1
-- Dumped by pg_dump version 9.5.1

SET statement_timeout = 0;
SET lock_timeout = 0;
SET client_encoding = 'UTF8';
SET standard_conforming_strings = on;
SET check_function_bodies = false;
SET client_min_messages = warning;
SET row_security = off;

--
-- Name: plpgsql; Type: EXTENSION; Schema: -; Owner: -
--

CREATE EXTENSION IF NOT EXISTS plpgsql WITH SCHEMA pg_catalog;


--
-- Name: EXTENSION plpgsql; Type: COMMENT; Schema: -; Owner: -
--

COMMENT ON EXTENSION plpgsql IS 'PL/pgSQL procedural language';


SET search_path = public, pg_catalog;

SET default_tablespace = '';

SET default_with_oids = false;

--
-- Name: ar_internal_metadata; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE ar_internal_metadata (
    key character varying NOT NULL,
    value character varying,
    created_at timestamp without time zone NOT NULL,
    updated_at timestamp without time zone NOT NULL
);


--
-- Name: exchange_rates; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE exchange_rates (
    id integer NOT NULL,
    base character varying(255),
    currency character varying(255),
    rate numeric,
    date date,
    source character varying(255),
    created_at timestamp without time zone NOT NULL,
    updated_at timestamp without time zone NOT NULL
);


--
-- Name: exchange_rates_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE exchange_rates_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: exchange_rates_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE exchange_rates_id_seq OWNED BY exchange_rates.id;


--
-- Name: fuel_prices; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE fuel_prices (
    id integer NOT NULL,
    basic_fuel_price numeric,
    full_95_coast numeric,
    date date,
    source character varying(255),
    created_at timestamp without time zone NOT NULL,
    updated_at timestamp without time zone NOT NULL,
    exchange_rate numeric,
    crude_oil numeric,
    bfp numeric,
    fuel_tax numeric,
    customs_excise numeric,
    equalization_fund_levy numeric,
    raf numeric,
    transport_cost numeric,
    petroleum_products_levy numeric,
    wholesale_margin numeric,
    secondary_storage numeric,
    secondary_distribution numeric,
    retail_margin numeric,
    slate_levy numeric,
    delivery_cost numeric,
    dsml numeric
);


--
-- Name: fuel_prices_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE fuel_prices_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: fuel_prices_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE fuel_prices_id_seq OWNED BY fuel_prices.id;


--
-- Name: oil_prices; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE oil_prices (
    id integer NOT NULL,
    currency character varying(255),
    price numeric,
    date date,
    source character varying(255),
    created_at timestamp without time zone NOT NULL,
    updated_at timestamp without time zone NOT NULL
);


--
-- Name: oil_prices_id_seq; Type: SEQUENCE; Schema: public; Owner: -
--

CREATE SEQUENCE oil_prices_id_seq
    START WITH 1
    INCREMENT BY 1
    NO MINVALUE
    NO MAXVALUE
    CACHE 1;


--
-- Name: oil_prices_id_seq; Type: SEQUENCE OWNED BY; Schema: public; Owner: -
--

ALTER SEQUENCE oil_prices_id_seq OWNED BY oil_prices.id;


--
-- Name: schema_migrations; Type: TABLE; Schema: public; Owner: -
--

CREATE TABLE schema_migrations (
    version character varying NOT NULL
);


--
-- Name: id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY exchange_rates ALTER COLUMN id SET DEFAULT nextval('exchange_rates_id_seq'::regclass);


--
-- Name: id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY fuel_prices ALTER COLUMN id SET DEFAULT nextval('fuel_prices_id_seq'::regclass);


--
-- Name: id; Type: DEFAULT; Schema: public; Owner: -
--

ALTER TABLE ONLY oil_prices ALTER COLUMN id SET DEFAULT nextval('oil_prices_id_seq'::regclass);


--
-- Name: ar_internal_metadata_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY ar_internal_metadata
    ADD CONSTRAINT ar_internal_metadata_pkey PRIMARY KEY (key);


--
-- Name: exchange_rates_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY exchange_rates
    ADD CONSTRAINT exchange_rates_pkey PRIMARY KEY (id);


--
-- Name: fuel_prices_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY fuel_prices
    ADD CONSTRAINT fuel_prices_pkey PRIMARY KEY (id);


--
-- Name: oil_prices_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY oil_prices
    ADD CONSTRAINT oil_prices_pkey PRIMARY KEY (id);


--
-- Name: schema_migrations_pkey; Type: CONSTRAINT; Schema: public; Owner: -
--

ALTER TABLE ONLY schema_migrations
    ADD CONSTRAINT schema_migrations_pkey PRIMARY KEY (version);


--
-- Name: exchange_rate_index; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX exchange_rate_index ON exchange_rates USING btree (date DESC);


--
-- Name: fuel_price_index; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX fuel_price_index ON fuel_prices USING btree (date DESC);


--
-- Name: oil_price_index; Type: INDEX; Schema: public; Owner: -
--

CREATE INDEX oil_price_index ON oil_prices USING btree (date DESC);


--
-- PostgreSQL database dump complete
--

SET search_path TO "$user", public;

INSERT INTO schema_migrations (version) VALUES ('20161119105900'), ('20161223171359'), ('20170104201226');


