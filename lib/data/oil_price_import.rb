class Data::OilPriceImport < CsvImport
  VALID_HEADERS = %W(Currency Value Date Source)
end