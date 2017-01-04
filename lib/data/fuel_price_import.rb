STRUCTURE = { 'Date' => { type: 'Date', permitted: [] },
              'Full_95_Coast' => { type: 'Decimal', permitted: [] },
              'Basic_Fuel_Price' => { type: 'Decimal', permitted: [] },
              'Exchange_Rate' => { type: 'Decimal', permitted: [] },
              'Crude_Oil' => { type: 'Decimal', permitted: [] },
              'BFP' => { type: 'Decimal', permitted: [] },
              'Fuel_Tax' => { type: 'Decimal', permitted: [] },
              'Customs_Excise' => { type: 'Decimal', permitted: [] },
              'Equalization_Fund_Levy' => { type: 'Decimal', permitted: [] },
              'RAF' => { type: 'Decimal', permitted: [] },
              'Transport_Cost' => { type: 'Decimal', permitted: [] },
              'Petroleum_Products_Levy' => { type: 'Decimal', permitted: [] },
              'Wholesale_Margin' => { type: 'Decimal', permitted: [] },
              'Secondary_Storage' => { type: 'Decimal', permitted: [] },
              'Secondary_Distribution' => { type: 'Decimal', permitted: [] },
              'Retail_Margin' => { type: 'Decimal', permitted: [] },
              'Slate_Levy' => { type: 'Decimal', permitted: [] },
              'Delivery_Cost' => { type: 'Decimal', permitted: [] },
              'DSML' => { type: 'Decimal', permitted: [] },
              'Source' => { type: 'String', permitted: ['http://www.energy.gov.za/files/petroleum_frame.html'] },
              'Update' => { type: 'String', permitted: ['yes', 'no'] }}

VALID_HEADERS = STRUCTURE.keys

class Data::FuelPriceImport < Data::CsvImport
  def self.import(file)
    fuel_import = new
    result = fuel_import.import(file, VALID_HEADERS)

    if result[:valid] == true
      fuel_import.insert_data(result[:data])
      result[:valid]
    else
      result[:valid]
    end
  end

  def initialize
    super
  end

  def insert_data(data)

  end
end