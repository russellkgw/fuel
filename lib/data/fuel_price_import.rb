class Data::FuelPriceImport < CsvImport
  VALID_HEADERS = %W(Currency Base Full Date Source)

  def initialize
    @errors = []
  end

end